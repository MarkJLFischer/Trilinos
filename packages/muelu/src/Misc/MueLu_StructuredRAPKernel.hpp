// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// @HEADER

#ifndef MUELU_STRUCTUREDRAPKERNEL_HPP
#define MUELU_STRUCTUREDRAPKERNEL_HPP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Xpetra_MapFactory.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_VectorFactory.hpp>

namespace MueLu {
namespace Details {

/**
 * Specialized numeric kernel for two- and three-dimensional structured RAP.
 *
 * The implemented path requires two or three dimensions and order-zero interpolation.
 * Its numeric plan is generated from the detected fine stencil and sparse
 * graphs; no generic sparse RAP fallback is provided.
 */
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class StructuredRAPKernel {
 public:
  using Matrix = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

 private:
  using LO          = LocalOrdinal;
  using device_type = typename Node::device_type;
  using lo_view     = Kokkos::View<LocalOrdinal*, device_type>;
  using size_view   = Kokkos::View<size_t*, device_type>;

  KOKKOS_INLINE_FUNCTION
  static LocalOrdinal coarseAnchor(const LocalOrdinal coarse,
                                   const LocalOrdinal numFine,
                                   const LocalOrdinal numCoarse,
                                   const LocalOrdinal rate) {
    return coarse + 1 == numCoarse ? numFine - 1 : coarse * rate;
  }

  // Inclusive fine-coordinate range belonging to an order-zero coarse node.
  // Keep the midpoint tie on the lower coarse node, including the short end interval.
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<LO, 2> aggregateBounds(LO coarse, LO numFine,
                                            LO numCoarse, LO rate) {
    const LO anchor = coarseAnchor(coarse, numFine, numCoarse, rate);
    const LO previous = coarse == 0 ? LO(0) : coarseAnchor(coarse - 1, numFine, numCoarse, rate);
    const LO next = coarse + 1 == numCoarse ? numFine - 1
                                          : coarseAnchor(coarse + 1, numFine, numCoarse, rate);
    return {{coarse == 0 ? LO(0) : previous + (anchor - previous) / 2 + 1,
             coarse + 1 == numCoarse ? numFine - 1 : anchor + (next - anchor) / 2}};
  }

  static size_t boundaryClass(LO coarse, LO numCoarse,
                             size_t leftDepth, size_t rightDepth) {
    const size_t left = static_cast<size_t>(coarse);
    const size_t right = static_cast<size_t>(numCoarse - coarse - 1);
    return left < leftDepth ? left
                           : (right < rightDepth ? leftDepth + rightDepth - right : leftDepth);
  }

  template <class Representative>
  static LO registerPlanClass(std::vector<std::vector<int>>& signatures,
                             std::vector<Representative>& classes,
                             std::vector<int>& signature,
                             const Representative& representative) {
    const auto found = std::find(signatures.begin(), signatures.end(), signature);
    const LO id = static_cast<LO>(found - signatures.begin());
    if (found == signatures.end()) {
      signatures.push_back(std::move(signature));
      classes.push_back(representative);
    }
    return id;
  }

  // Keep host mirrors alive until the preparation fence, allowing asynchronous copies.
  template <class View>
  struct PlanArray {
    View device;
    typename View::host_mirror_type host;

    PlanArray(const char* label, size_t size)
        : device(Kokkos::ViewAllocateWithoutInitializing(label), size),
          host(Kokkos::create_mirror_view(device)) {}

    View copyToDevice(const typename device_type::execution_space& executionSpace) const {
      Kokkos::deep_copy(executionSpace, device, host);
      return device;
    }
  };

  KOKKOS_INLINE_FUNCTION
  static LocalOrdinal constantInterpolationCoarse(
      const LocalOrdinal fine,
      const LocalOrdinal numFine,
      const LocalOrdinal rate) {
    LocalOrdinal endRate = (numFine - 1) % rate;
    if (endRate == 0)
      endRate = rate;

    LocalOrdinal coarse = fine / rate;
    const LocalOrdinal remainder = fine % rate;
    const LocalOrdinal activeRate =
        fine < numFine - endRate ? rate : endRate;
    if (remainder > activeRate / 2)
      ++coarse;
    return coarse;
  }

 public:
  template <class FineStencil>
  static void Compute(const Matrix& A,
                      const Matrix& P,
                      Matrix& Ac,
                      const FineStencil& fineStencil,
                      const int interpolationOrder,
                      const Teuchos::Array<LocalOrdinal>& lFineNodesPerDim,
                      const Teuchos::Array<LocalOrdinal>& lCoarseNodesPerDim,
                      const Teuchos::Array<int>& structuredCoarseningRate,
                      const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
    using SC              = Scalar;
    using execution_space = typename device_type::execution_space;
    using range_policy    = Kokkos::RangePolicy<execution_space,
                                                Kokkos::IndexType<size_t>>;
    const std::string timerPrefix = "MueLu: StructuredRAPKernel: ";

    const int numDimensions = fineStencil.numDimensions;
    const LO dofsPerNode     = fineStencil.dofsPerNode;

    Kokkos::Array<LO, 3> localFineNodes{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    Kokkos::Array<LO, 3> localCoarseNodes{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    Kokkos::Array<LO, 3> coarseningRate{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    for (int dim = 0; dim < numDimensions; ++dim) {
      localFineNodes[dim] = lFineNodesPerDim[dim];
      localCoarseNodes[dim] = lCoarseNodesPerDim[dim];
      coarseningRate[dim] = static_cast<LO>(
          structuredCoarseningRate.size() == 1
              ? structuredCoarseningRate[0]
              : structuredCoarseningRate[dim]);
    }

    execution_space executionSpace;
    bool structuredDimensionsMatch = true;
    for (int dim = 0; dim < numDimensions && structuredDimensionsMatch; ++dim) {
      const LO expectedCoarseNodes =
          localFineNodes[dim] == 1
              ? LO(1)
              : static_cast<LO>((localFineNodes[dim] - 2) /
                                    coarseningRate[dim] +
                                2);
      structuredDimensionsMatch =
          localCoarseNodes[dim] == expectedCoarseNodes;
    }

    const size_t expectedFineRows =
        static_cast<size_t>(localFineNodes[0]) *
        static_cast<size_t>(localFineNodes[1]) *
        static_cast<size_t>(localFineNodes[2]) *
        static_cast<size_t>(dofsPerNode);
    const size_t expectedCoarseRows =
        static_cast<size_t>(localCoarseNodes[0]) *
        static_cast<size_t>(localCoarseNodes[1]) *
        static_cast<size_t>(localCoarseNodes[2]) *
        static_cast<size_t>(dofsPerNode);
    const bool distributed = A.getRowMap()->getComm()->getSize() > 1;
    const bool useDetectedStencil =
        interpolationOrder == 0 &&
        (numDimensions == 2 || numDimensions == 3) &&
        structuredDimensionsMatch &&
        A.GetFixedBlockSize() == dofsPerNode &&
        A.getRowMap()->getLocalNumElements() == expectedFineRows &&
        Ac.getRowMap()->getLocalNumElements() == expectedCoarseRows;

    if (!useDetectedStencil)
      throw std::runtime_error(
          "StructuredRAPKernel supports only 2D or 3D order-zero structured RAP.");

    const SC zero  = Teuchos::ScalarTraits<SC>::zero();
    const auto localA = A.getLocalMatrixDevice();
    auto localAc      = Ac.getLocalMatrixDevice();

    // Order-zero interpolation has exactly one unit entry per fine row.  Only
    // off-rank A columns need communication: local columns can use the same
    // structured coordinate calculation as the serial path.  Reuse A's graph
    // importer and exchange one coarse GID per halo row instead of constructing
    // and synchronizing a vector over the entire A column map.
    using GOVector = Xpetra::Vector<GlobalOrdinal, LO, GlobalOrdinal, Node>;
    Teuchos::RCP<const Xpetra::Map<LO, GlobalOrdinal, Node>> remoteFineMap;
    Teuchos::RCP<GOVector> remoteCoarseGids;
    Teuchos::ArrayRCP<const GlobalOrdinal> remoteCoarseGidData;
    if (distributed) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          !P.getRowMap()->isSameAs(*A.getRowMap()), std::runtime_error,
          "StructuredRAPKernel requires A and P to have identical row maps.");

      const auto graphImporter = A.getCrsGraph()->getImporter();
      TEUCHOS_TEST_FOR_EXCEPTION(
          graphImporter.is_null(), std::runtime_error,
          "StructuredRAPKernel requires an A graph importer in parallel.");
      const auto remoteLids = graphImporter->getRemoteLIDs();
      Teuchos::Array<GlobalOrdinal> remoteFineGids(remoteLids.size());
      for (size_t i = 0; i < static_cast<size_t>(remoteLids.size()); ++i)
        remoteFineGids[i] = A.getColMap()->getGlobalElement(remoteLids[i]);
      remoteFineMap = Xpetra::MapFactory<LO, GlobalOrdinal, Node>::Build(
          A.getRowMap()->lib(),
          Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid(),
          remoteFineGids(), A.getRowMap()->getIndexBase(),
          A.getRowMap()->getComm());
      const auto remoteImporter =
          graphImporter->createRemoteOnlyImport(remoteFineMap);

      const auto localP = P.getLocalMatrixDevice();
      auto ownedCoarseGids =
          Xpetra::VectorFactory<GlobalOrdinal, LO, GlobalOrdinal, Node>::Build(
              A.getRowMap(), false);
      auto ownedData = ownedCoarseGids->getLocalViewDevice(
          Tpetra::Access::OverwriteAll);
      const auto coarseColumnMap = P.getColMap()->getLocalMap();
      const auto exportLidsHost = remoteImporter->getExportLIDs();
      PlanArray<lo_view> exportLids(
          "StructuredRAP: halo export LIDs", exportLidsHost.size());
      for (size_t i = 0; i < static_cast<size_t>(exportLidsHost.size()); ++i)
        exportLids.host(i) = exportLidsHost[i];
      const auto exportLidsDevice = exportLids.copyToDevice(executionSpace);
      size_t invalidInterpolationRows = 0;
      Kokkos::parallel_reduce(
          "StructuredRAP: build halo coarse target GIDs",
          range_policy(executionSpace, 0,
                       static_cast<size_t>(exportLidsHost.size())),
          KOKKOS_LAMBDA(const size_t exportIndex, size_t& invalid) {
            const LO row = exportLidsDevice(exportIndex);
            const size_t rowBegin = localP.graph.row_map(row);
            const size_t rowEnd   = localP.graph.row_map(row + 1);
            if (rowEnd - rowBegin != size_t(1)) {
              ++invalid;
              return;
            }
            ownedData(row, 0) = coarseColumnMap.getGlobalElement(
                localP.graph.entries(rowBegin));
          },
          invalidInterpolationRows);
      executionSpace.fence("StructuredRAP: coarse target GIDs ready");
      TEUCHOS_TEST_FOR_EXCEPTION(
          invalidInterpolationRows != 0, std::runtime_error,
          "StructuredRAPKernel requires exactly one interpolation entry per fine row.");

      remoteCoarseGids =
          Xpetra::VectorFactory<GlobalOrdinal, LO, GlobalOrdinal, Node>::Build(
              remoteFineMap, false);
      remoteCoarseGids->doImport(
          *ownedCoarseGids, *remoteImporter, Xpetra::INSERT);
      remoteCoarseGidData = remoteCoarseGids->getData(0);
    }

    const size_t numFineStencilOffsets = fineStencil.stencilOffsets.size();
    int minStencilOffsetX = fineStencil.stencilOffsets[0].x;
    int maxStencilOffsetX = fineStencil.stencilOffsets[0].x;
    int minStencilOffsetY = fineStencil.stencilOffsets[0].y;
    int maxStencilOffsetY = fineStencil.stencilOffsets[0].y;
    int minStencilOffsetZ = fineStencil.stencilOffsets[0].z;
    int maxStencilOffsetZ = fineStencil.stencilOffsets[0].z;
    for (size_t offset = 1; offset < numFineStencilOffsets; ++offset) {
      minStencilOffsetX = std::min(
          minStencilOffsetX, fineStencil.stencilOffsets[offset].x);
      maxStencilOffsetX = std::max(
          maxStencilOffsetX, fineStencil.stencilOffsets[offset].x);
      minStencilOffsetY = std::min(
          minStencilOffsetY, fineStencil.stencilOffsets[offset].y);
      maxStencilOffsetY = std::max(
          maxStencilOffsetY, fineStencil.stencilOffsets[offset].y);
      minStencilOffsetZ = std::min(
          minStencilOffsetZ, fineStencil.stencilOffsets[offset].z);
      maxStencilOffsetZ = std::max(
          maxStencilOffsetZ, fineStencil.stencilOffsets[offset].z);
    }
    const size_t numCoarseNodes =
        static_cast<size_t>(localCoarseNodes[0]) *
        static_cast<size_t>(localCoarseNodes[1]) *
        static_cast<size_t>(localCoarseNodes[2]);
    const size_t numDofs = static_cast<size_t>(dofsPerNode);
    const size_t invalidStencilNodeOrdinal =
        static_cast<size_t>(-1);

    lo_view coarseXAggregatePlanClass;
    lo_view coarseYAggregatePlanClass;
    lo_view coarseZAggregatePlanClass;
    size_view reverseContributionBegins;
    size_view reverseContributionCounts;
    size_view reverseAValueOffsets;
    size_t numXAggregatePlanClasses = 0;
    size_t numYAggregatePlanClasses = 0;
    size_t numZAggregatePlanClasses = 0;
    size_t maxAcEntriesPerRow = 0;
    {
      Teuchos::TimeMonitor timer(
          *Teuchos::TimeMonitor::getNewTimer(
              timerPrefix + "prepare numeric entry plan"));

      struct AxisPlanClass {};

      auto buildAxisPlanClasses = [&](const LO numFine,
                                      const LO numCoarse,
                                      const LO rate,
                                      const int minOffset,
                                      const int maxOffset,
                                      const size_t leftDepth,
                                      const size_t rightDepth,
                                      Teuchos::Array<LO>& classByFine,
                                      std::vector<AxisPlanClass>& classes) {
        std::vector<std::vector<int>> signatures;
        for (LO fine = 0; fine < numFine; ++fine) {
          const LO coarse = constantInterpolationCoarse(
              fine, numFine, rate);
          std::vector<int> signature;
          signature.reserve(static_cast<size_t>(maxOffset - minOffset + 2));
          signature.push_back(static_cast<int>(
              boundaryClass(coarse, numCoarse, leftDepth, rightDepth)));
          for (int offset = minOffset; offset <= maxOffset; ++offset) {
            const LO neighbor = fine + static_cast<LO>(offset);
            if (neighbor < LO(0) || neighbor >= numFine) {
              signature.push_back(0);
              signature.push_back(0);
            } else {
              const LO neighborCoarse = constantInterpolationCoarse(
                  neighbor, numFine, rate);
              signature.push_back(1);
              signature.push_back(static_cast<int>(neighborCoarse - coarse));
            }
          }

          classByFine[static_cast<size_t>(fine)] = registerPlanClass(
              signatures, classes, signature, AxisPlanClass{});
        }
      };

      Teuchos::Array<LO> fineXPlanClassHost(
          static_cast<size_t>(localFineNodes[0]));
      Teuchos::Array<LO> fineYPlanClassHost(
          static_cast<size_t>(localFineNodes[1]));
      Teuchos::Array<LO> fineZPlanClassHost(
          static_cast<size_t>(localFineNodes[2]));
      std::vector<AxisPlanClass> xPlanClasses;
      std::vector<AxisPlanClass> yPlanClasses;
      std::vector<AxisPlanClass> zPlanClasses;

      const auto aRowMapHost = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), localA.graph.row_map);
      const auto aEntriesHost = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), localA.graph.entries);
      const auto acRowMapHost = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), localAc.graph.row_map);
      const auto acEntriesHost = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), localAc.graph.entries);

      std::vector<size_t> maxAcEntriesPerDof(numDofs, size_t(0));
      std::vector<size_t> maxAcRepresentativeRows(
          numDofs, invalidStencilNodeOrdinal);
      for (size_t coarseNode = 0; coarseNode < numCoarseNodes;
           ++coarseNode) {
        for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
          const size_t row = coarseNode * numDofs + rowDof;
          const size_t rowEntries =
              acRowMapHost(row + size_t(1)) - acRowMapHost(row);
          if (rowEntries > maxAcEntriesPerDof[rowDof]) {
            maxAcEntriesPerDof[rowDof] = rowEntries;
            maxAcRepresentativeRows[rowDof] = row;
          }
        }
      }

      // Infer each axis's coarse boundary depth from the longest rows.
      // If any representative is not interior, keep every coordinate distinct.
      size_t leftBoundaryDepth[3], rightBoundaryDepth[3];
      if (distributed) {
        // Remote column LIDs do not encode local structured coordinates, so
        // infer reusable boundary classes from the detected stencil radius
        // instead.  Plans are rank-local, allowing physical and interprocess
        // boundaries to use different representatives without assigning a
        // distinct class to every coordinate.
        leftBoundaryDepth[0]  = static_cast<size_t>(std::max(0, -minStencilOffsetX));
        rightBoundaryDepth[0] = static_cast<size_t>(std::max(0, maxStencilOffsetX));
        leftBoundaryDepth[1]  = static_cast<size_t>(std::max(0, -minStencilOffsetY));
        rightBoundaryDepth[1] = static_cast<size_t>(std::max(0, maxStencilOffsetY));
        leftBoundaryDepth[2]  = static_cast<size_t>(std::max(0, -minStencilOffsetZ));
        rightBoundaryDepth[2] = static_cast<size_t>(std::max(0, maxStencilOffsetZ));
      } else {
        for (int dim = 0; dim < 3; ++dim) {
          auto coordinate = [&](size_t node) {
            if (dim == 0)
              return static_cast<LO>(node % localCoarseNodes[0]);
            node /= localCoarseNodes[0];
            return static_cast<LO>(dim == 1 ? node % localCoarseNodes[1]
                                           : node / localCoarseNodes[1]);
          };
          int minOffset = 0, maxOffset = 0;
          for (const size_t row : maxAcRepresentativeRows) {
            const LO source = coordinate(row / numDofs);
            for (size_t entry = acRowMapHost(row); entry < acRowMapHost(row + 1); ++entry) {
              const int offset = static_cast<int>(coordinate(acEntriesHost(entry) / dofsPerNode) - source);
              minOffset = std::min(minOffset, offset);
              maxOffset = std::max(maxOffset, offset);
            }
          }
          leftBoundaryDepth[dim] = static_cast<size_t>(std::max(0, -minOffset));
          rightBoundaryDepth[dim] = static_cast<size_t>(std::max(0, maxOffset));
          for (const size_t row : maxAcRepresentativeRows) {
            const size_t source = static_cast<size_t>(coordinate(row / numDofs));
            if (source < leftBoundaryDepth[dim] ||
                static_cast<size_t>(localCoarseNodes[dim]) - source - 1 < rightBoundaryDepth[dim]) {
              leftBoundaryDepth[dim] = static_cast<size_t>(localCoarseNodes[dim]);
              rightBoundaryDepth[dim] = 0;
              break;
            }
          }
        }
      }

      buildAxisPlanClasses(
          localFineNodes[0], localCoarseNodes[0], coarseningRate[0],
          minStencilOffsetX, maxStencilOffsetX, leftBoundaryDepth[0],
          rightBoundaryDepth[0], fineXPlanClassHost, xPlanClasses);
      buildAxisPlanClasses(
          localFineNodes[1], localCoarseNodes[1], coarseningRate[1],
          minStencilOffsetY, maxStencilOffsetY, leftBoundaryDepth[1],
          rightBoundaryDepth[1], fineYPlanClassHost, yPlanClasses);
      buildAxisPlanClasses(
          localFineNodes[2], localCoarseNodes[2], coarseningRate[2],
          minStencilOffsetZ, maxStencilOffsetZ, leftBoundaryDepth[2],
          rightBoundaryDepth[2], fineZPlanClassHost, zPlanClasses);
      struct AxisAggregatePlanClass {
        LO representativeCoarse;
        LO fineBegin;
        LO fineEnd;
      };
      auto buildAxisAggregatePlanClasses = [&](
          const LO numFine, const LO numCoarse, const LO rate,
          const size_t leftDepth, const size_t rightDepth,
          const Teuchos::Array<LO>& fineClassByCoordinate,
          Teuchos::Array<LO>& classByCoarse,
          std::vector<AxisAggregatePlanClass>& classes) {
        std::vector<std::vector<int>> signatures;
        for (LO coarse = 0; coarse < numCoarse; ++coarse) {
          const auto bounds = aggregateBounds(coarse, numFine, numCoarse, rate);
          const LO fineBegin = bounds[0], fineEnd = bounds[1];
          std::vector<int> signature;
          signature.reserve(
              static_cast<size_t>(fineEnd - fineBegin + LO(2)));
          signature.push_back(static_cast<int>(
              boundaryClass(coarse, numCoarse, leftDepth, rightDepth)));
          for (LO fine = fineBegin; fine <= fineEnd; ++fine)
            signature.push_back(static_cast<int>(
                fineClassByCoordinate[static_cast<size_t>(fine)]));

          classByCoarse[static_cast<size_t>(coarse)] = registerPlanClass(
              signatures, classes, signature,
              AxisAggregatePlanClass{coarse, fineBegin, fineEnd});
        }
      };

      Teuchos::Array<LO> coarseXAggregatePlanClassHost(
          static_cast<size_t>(localCoarseNodes[0]));
      Teuchos::Array<LO> coarseYAggregatePlanClassHost(
          static_cast<size_t>(localCoarseNodes[1]));
      Teuchos::Array<LO> coarseZAggregatePlanClassHost(
          static_cast<size_t>(localCoarseNodes[2]));
      std::vector<AxisAggregatePlanClass> xAggregatePlanClasses;
      std::vector<AxisAggregatePlanClass> yAggregatePlanClasses;
      std::vector<AxisAggregatePlanClass> zAggregatePlanClasses;
      buildAxisAggregatePlanClasses(
          localFineNodes[0], localCoarseNodes[0], coarseningRate[0],
          leftBoundaryDepth[0], rightBoundaryDepth[0],
          fineXPlanClassHost, coarseXAggregatePlanClassHost,
          xAggregatePlanClasses);
      buildAxisAggregatePlanClasses(
          localFineNodes[1], localCoarseNodes[1], coarseningRate[1],
          leftBoundaryDepth[1], rightBoundaryDepth[1],
          fineYPlanClassHost, coarseYAggregatePlanClassHost,
          yAggregatePlanClasses);
      buildAxisAggregatePlanClasses(
          localFineNodes[2], localCoarseNodes[2], coarseningRate[2],
          leftBoundaryDepth[2], rightBoundaryDepth[2],
          fineZPlanClassHost, coarseZAggregatePlanClassHost,
          zAggregatePlanClasses);
      numXAggregatePlanClasses = xAggregatePlanClasses.size();
      numYAggregatePlanClasses = yAggregatePlanClasses.size();
      numZAggregatePlanClasses = zAggregatePlanClasses.size();

      for (const size_t rowEntries : maxAcEntriesPerDof)
        maxAcEntriesPerRow =
            std::max(maxAcEntriesPerRow, rowEntries);

      PlanArray<lo_view> xClasses("StructuredRAP: coarse-X aggregate plan class", localCoarseNodes[0]);
      PlanArray<lo_view> yClasses("StructuredRAP: coarse-Y aggregate plan class", localCoarseNodes[1]);
      PlanArray<lo_view> zClasses("StructuredRAP: coarse-Z aggregate plan class", localCoarseNodes[2]);
      for (size_t i = 0; i < static_cast<size_t>(localCoarseNodes[0]); ++i)
        xClasses.host(i) = coarseXAggregatePlanClassHost[i];
      for (size_t i = 0; i < static_cast<size_t>(localCoarseNodes[1]); ++i)
        yClasses.host(i) = coarseYAggregatePlanClassHost[i];
      for (size_t i = 0; i < static_cast<size_t>(localCoarseNodes[2]); ++i)
        zClasses.host(i) = coarseZAggregatePlanClassHost[i];
      coarseXAggregatePlanClass = xClasses.copyToDevice(executionSpace);
      coarseYAggregatePlanClass = yClasses.copyToDevice(executionSpace);
      coarseZAggregatePlanClass = zClasses.copyToDevice(executionSpace);

      auto coarseTargetColumn = [&](const LO fineColumn) {
        const GlobalOrdinal fineColumnGid =
            A.getColMap()->getGlobalElement(fineColumn);
        const LO ownedFineColumn =
            A.getRowMap()->getLocalElement(fineColumnGid);
        GlobalOrdinal targetCoarseGid;
        if (ownedFineColumn != Teuchos::OrdinalTraits<LO>::invalid()) {
          const LO columnNode = ownedFineColumn / dofsPerNode;
          const LO columnDof = ownedFineColumn % dofsPerNode;
          const LO columnX = columnNode % localFineNodes[0];
          const LO columnYZ = columnNode / localFineNodes[0];
          const LO columnY = columnYZ % localFineNodes[1];
          const LO columnZ = columnYZ / localFineNodes[1];
          const LO targetX = constantInterpolationCoarse(
              columnX, localFineNodes[0], coarseningRate[0]);
          const LO targetY = constantInterpolationCoarse(
              columnY, localFineNodes[1], coarseningRate[1]);
          const LO targetZ = constantInterpolationCoarse(
              columnZ, localFineNodes[2], coarseningRate[2]);
          const LO targetRow = Teuchos::as<LO>(
              (static_cast<size_t>(targetX) +
               static_cast<size_t>(localCoarseNodes[0]) *
                   (static_cast<size_t>(targetY) +
                    static_cast<size_t>(localCoarseNodes[1]) * targetZ)) *
                  numDofs +
              static_cast<size_t>(columnDof));
          targetCoarseGid = Ac.getRowMap()->getGlobalElement(targetRow);
        } else {
          const LO remoteFineColumn =
              remoteFineMap->getLocalElement(fineColumnGid);
          TEUCHOS_TEST_FOR_EXCEPTION(
              remoteFineColumn == Teuchos::OrdinalTraits<LO>::invalid(),
              std::runtime_error,
              "StructuredRAPKernel remote fine column is absent from the halo map.");
          targetCoarseGid = remoteCoarseGidData[remoteFineColumn];
        }
        const LO targetColumn =
            Ac.getColMap()->getLocalElement(targetCoarseGid);
        TEUCHOS_TEST_FOR_EXCEPTION(
            targetColumn == Teuchos::OrdinalTraits<LO>::invalid(),
            std::runtime_error,
            "StructuredRAPKernel interpolation target is absent from the coarse column map.");
        return targetColumn;
      };

      const size_t reversePlanRowCount =
          numXAggregatePlanClasses * numYAggregatePlanClasses *
          numZAggregatePlanClasses * numDofs;
      const size_t reverseBucketCount =
          reversePlanRowCount * maxAcEntriesPerRow;
      std::vector<std::vector<size_t>> reverseBuckets(reverseBucketCount);
      for (size_t zClass = 0; zClass < numZAggregatePlanClasses; ++zClass) {
        const auto& zAggregate = zAggregatePlanClasses[zClass];
        const LO coarseZ = zAggregate.representativeCoarse;
        const size_t fineZBegin = static_cast<size_t>(zAggregate.fineBegin);
        const size_t fineZEnd = static_cast<size_t>(zAggregate.fineEnd);
        for (size_t yClass = 0; yClass < numYAggregatePlanClasses; ++yClass) {
          const auto& yAggregate = yAggregatePlanClasses[yClass];
          const LO coarseY = yAggregate.representativeCoarse;
          const size_t fineYBegin = static_cast<size_t>(yAggregate.fineBegin);
          const size_t fineYEnd = static_cast<size_t>(yAggregate.fineEnd);
          for (size_t xClass = 0; xClass < numXAggregatePlanClasses; ++xClass) {
            const auto& xAggregate = xAggregatePlanClasses[xClass];
            const LO coarseX = xAggregate.representativeCoarse;
            const size_t fineXBegin = static_cast<size_t>(xAggregate.fineBegin);
            const size_t fineXEnd = static_cast<size_t>(xAggregate.fineEnd);
            const size_t coarseNode =
                static_cast<size_t>(coarseX) +
                static_cast<size_t>(localCoarseNodes[0]) *
                    (static_cast<size_t>(coarseY) +
                     static_cast<size_t>(localCoarseNodes[1]) * coarseZ);
            const size_t firstFineNode =
                fineXBegin + static_cast<size_t>(localFineNodes[0]) *
                    (fineYBegin + static_cast<size_t>(localFineNodes[1]) * fineZBegin);
            for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
              const size_t reverseRow =
                  ((zClass * numYAggregatePlanClasses + yClass) *
                       numXAggregatePlanClasses + xClass) * numDofs + rowDof;
              const size_t firstAValue =
                  aRowMapHost(firstFineNode * numDofs + rowDof);
              const size_t acRow = coarseNode * numDofs + rowDof;
              const size_t acRowBegin = acRowMapHost(acRow);
              const size_t acRowEnd = acRowMapHost(acRow + 1);
              for (size_t fineZ = fineZBegin; fineZ <= fineZEnd; ++fineZ) {
                for (size_t fineY = fineYBegin; fineY <= fineYEnd; ++fineY) {
                  for (size_t fineX = fineXBegin; fineX <= fineXEnd; ++fineX) {
                    const size_t sourceFineNode =
                        fineX + static_cast<size_t>(localFineNodes[0]) *
                            (fineY + static_cast<size_t>(localFineNodes[1]) * fineZ);
                    const size_t aRowBegin =
                        aRowMapHost(sourceFineNode * numDofs + rowDof);
                    const size_t aRowEnd =
                        aRowMapHost(sourceFineNode * numDofs + rowDof + 1);
                    for (size_t aEntry = aRowBegin; aEntry < aRowEnd; ++aEntry) {
                      const LO targetColumn =
                          coarseTargetColumn(aEntriesHost(aEntry));
                      for (size_t acEntry = acRowBegin; acEntry < acRowEnd;
                           ++acEntry) {
                        if (acEntriesHost(acEntry) == targetColumn) {
                          const size_t bucket =
                              reverseRow * maxAcEntriesPerRow +
                              acEntry - acRowBegin;
                          // Rows with the same aggregate plan class have the same CSR layout.
                          reverseBuckets[bucket].push_back(
                              aEntry - firstAValue);
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      size_t numContributions = 0;
      for (const auto& bucket : reverseBuckets)
        numContributions += bucket.size();
      PlanArray<size_view> begins("StructuredRAP: reverse contribution begins", reverseBucketCount);
      PlanArray<size_view> counts("StructuredRAP: reverse contribution counts", reverseBucketCount);
      PlanArray<size_view> aValueOffsets(
          "StructuredRAP: reverse A value offsets", numContributions);
      size_t contribution = 0;
      for (size_t bucket = 0; bucket < reverseBucketCount; ++bucket) {
        begins.host(bucket) = contribution;
        counts.host(bucket) = reverseBuckets[bucket].size();
        for (const size_t offset : reverseBuckets[bucket])
          aValueOffsets.host(contribution++) = offset;
      }
      reverseContributionBegins = begins.copyToDevice(executionSpace);
      reverseContributionCounts = counts.copyToDevice(executionSpace);
      reverseAValueOffsets = aValueOffsets.copyToDevice(executionSpace);

      executionSpace.fence("StructuredRAP: numeric entry plan prepared");
    }

    {
      Teuchos::TimeMonitor timer(
          *Teuchos::TimeMonitor::getNewTimer(
              timerPrefix +
              "numeric (constant stencil)"));
      range_policy policy(executionSpace, 0, numCoarseNodes);
      Kokkos::parallel_for(
          "StructuredRAP: constant stencil P^T A P",
          policy,
          KOKKOS_LAMBDA(const size_t coarseNodeIndex) {
            const SC* KOKKOS_RESTRICT aValues = localA.values.data();
            SC* KOKKOS_RESTRICT acValues = localAc.values.data();
            const size_t* KOKKOS_RESTRICT aValueOffsets =
                reverseAValueOffsets.data();
            const LO coarseNode = static_cast<LO>(coarseNodeIndex);
            const LO coarseX = coarseNode % localCoarseNodes[0];
            const LO coarseYZ = coarseNode / localCoarseNodes[0];
            const LO coarseY = coarseYZ % localCoarseNodes[1];
            const LO coarseZ = coarseYZ / localCoarseNodes[1];
            const size_t firstCoarseRow = coarseNodeIndex * numDofs;

            const LO fineXBegin = aggregateBounds(
                coarseX, localFineNodes[0], localCoarseNodes[0], coarseningRate[0])[0];
            const LO fineYBegin = aggregateBounds(
                coarseY, localFineNodes[1], localCoarseNodes[1], coarseningRate[1])[0];
            const LO fineZBegin = aggregateBounds(
                coarseZ, localFineNodes[2], localCoarseNodes[2], coarseningRate[2])[0];

            const size_t firstFineNode =
                static_cast<size_t>(fineXBegin) +
                static_cast<size_t>(localFineNodes[0]) *
                    (static_cast<size_t>(fineYBegin) +
                     static_cast<size_t>(localFineNodes[1]) * fineZBegin);
            const size_t aggregateXClass = coarseXAggregatePlanClass(coarseX);
            const size_t aggregateYClass = coarseYAggregatePlanClass(coarseY);
            const size_t aggregateZClass = coarseZAggregatePlanClass(coarseZ);
            const size_t reversePlanNodeBegin =
                ((aggregateZClass * numYAggregatePlanClasses + aggregateYClass) *
                     numXAggregatePlanClasses + aggregateXClass) * numDofs;

            for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
              const size_t acRow = firstCoarseRow + rowDof;
              const size_t acRowBegin = localAc.graph.row_map(acRow);
              const size_t acRowEnd = localAc.graph.row_map(acRow + size_t(1));
              const size_t reversePlanRow = reversePlanNodeBegin + rowDof;
              const size_t firstAValue =
                  localA.graph.row_map(firstFineNode * numDofs + rowDof);
              for (size_t acOffset = 0; acOffset < acRowEnd - acRowBegin; ++acOffset) {
                const size_t bucket =
                    reversePlanRow * maxAcEntriesPerRow + acOffset;
                const size_t contributionBegin = reverseContributionBegins(bucket);
                const size_t contributionEnd =
                    contributionBegin + reverseContributionCounts(bucket);
                SC sum0 = zero, sum1 = zero, sum2 = zero, sum3 = zero;
                size_t contribution = contributionBegin;
                for (; contribution + 3 < contributionEnd; contribution += 4) {
                  sum0 += aValues[firstAValue + aValueOffsets[contribution]];
                  sum1 += aValues[firstAValue + aValueOffsets[contribution + 1]];
                  sum2 += aValues[firstAValue + aValueOffsets[contribution + 2]];
                  sum3 += aValues[firstAValue + aValueOffsets[contribution + 3]];
                }
                SC sum = (sum0 + sum1) + (sum2 + sum3);
                for (; contribution < contributionEnd; ++contribution)
                  sum += aValues[firstAValue + aValueOffsets[contribution]];
                acValues[acRowBegin + acOffset] = sum;
              }
            }
          });
      executionSpace.fence(
          "StructuredRAP: constant stencil product complete");
    }

    if (!Ac.isFillComplete()) {
      Teuchos::TimeMonitor timer(
          *Teuchos::TimeMonitor::getNewTimer(timerPrefix +
                                         "fillComplete"));
      Ac.fillComplete(P.getDomainMap(), P.getDomainMap(), params);
    }
  }
};

}  // namespace Details
}  // namespace MueLu

#endif  // MUELU_STRUCTUREDRAPKERNEL_HPP
