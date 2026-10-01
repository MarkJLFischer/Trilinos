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
#include <vector>

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Xpetra_Matrix.hpp>

namespace MueLu {
namespace Details {

/**
 * Specialized numeric kernel for single-rank, two-dimensional structured RAP.
 *
 * The implemented path requires two dimensions and order-zero interpolation.
 * Its numeric plan is generated from the detected fine stencil and sparse
 * graphs; no generic sparse RAP fallback is provided.
 */
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class StructuredRAPKernel {
 public:
  using Matrix = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

 private:
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
    using LO              = LocalOrdinal;
    using SC              = Scalar;
    using execution_space = typename device_type::execution_space;
    using range_policy    = Kokkos::RangePolicy<execution_space,
                                                Kokkos::IndexType<size_t>>;
    static constexpr size_t coarseNodesPerTile = 8;

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
        static_cast<size_t>(dofsPerNode);
    const size_t expectedCoarseRows =
        static_cast<size_t>(localCoarseNodes[0]) *
        static_cast<size_t>(localCoarseNodes[1]) *
        static_cast<size_t>(dofsPerNode);
    const bool useDetectedStencil =
        interpolationOrder == 0 &&
        numDimensions == 2 &&
        structuredDimensionsMatch &&
        A.GetFixedBlockSize() == dofsPerNode &&
        A.getRowMap()->getComm()->getSize() == 1 &&
        A.getColMap()->isSameAs(*A.getRowMap()) &&
        Ac.getColMap()->isSameAs(*Ac.getRowMap()) &&
        A.getRowMap()->getLocalNumElements() == expectedFineRows &&
        Ac.getRowMap()->getLocalNumElements() == expectedCoarseRows;

    if (useDetectedStencil) {
        const SC zero  = Teuchos::ScalarTraits<SC>::zero();
        const auto localA = A.getLocalMatrixDevice();
        auto localAc      = Ac.getLocalMatrixDevice();

        const size_t numFineStencilOffsets = fineStencil.stencilOffsets.size();
        const size_t numFineStencilEntries = fineStencil.entries.size();
        int minStencilOffsetX = fineStencil.stencilOffsets[0].x;
        int maxStencilOffsetX = fineStencil.stencilOffsets[0].x;
        int minStencilOffsetY = fineStencil.stencilOffsets[0].y;
        int maxStencilOffsetY = fineStencil.stencilOffsets[0].y;
        for (size_t offset = 1; offset < numFineStencilOffsets; ++offset) {
          minStencilOffsetX = std::min(
              minStencilOffsetX, fineStencil.stencilOffsets[offset].x);
          maxStencilOffsetX = std::max(
              maxStencilOffsetX, fineStencil.stencilOffsets[offset].x);
          minStencilOffsetY = std::min(
              minStencilOffsetY, fineStencil.stencilOffsets[offset].y);
          maxStencilOffsetY = std::max(
              maxStencilOffsetY, fineStencil.stencilOffsets[offset].y);
        }
        const size_t numCoarseNodes =
            static_cast<size_t>(localCoarseNodes[0]) *
            static_cast<size_t>(localCoarseNodes[1]);
        const size_t numScalarDofs = static_cast<size_t>(dofsPerNode);
        const size_t invalidStencilNodeOrdinal =
            static_cast<size_t>(-1);

        lo_view coarseXAggregatePlanClass;
        lo_view coarseYAggregatePlanClass;
        size_view reverseContributionBegins;
        size_view reverseContributionCounts;
        size_view reverseFineNodeOffsets;
        size_view reverseAEntryOrdinals;
        size_view fineARowBegins;
        size_t numXPlanClasses = 0;
        size_t numYPlanClasses = 0;
        size_t numXAggregatePlanClasses = 0;
        size_t numYAggregatePlanClasses = 0;
        size_t maxAcEntriesPerRow = 0;
        size_t maxAEntriesPerRow = 0;
        size_t maxFineNodesPerAggregate = 0;
        {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(
                  timerPrefix + "prepare numeric entry plan"));

          struct AxisPlanClass {
            LO representativeFine;
            LO representativeCoarse;
          };

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
              const size_t coarseIndex = static_cast<size_t>(coarse);
              const size_t rightDistance =
                  static_cast<size_t>(numCoarse - coarse - LO(1));
              const size_t boundaryClass =
                  coarseIndex < leftDepth
                      ? coarseIndex
                      : (rightDistance < rightDepth
                             ? leftDepth + rightDepth - rightDistance
                             : leftDepth);

              std::vector<int> signature;
              signature.reserve(static_cast<size_t>(maxOffset - minOffset + 2));
              signature.push_back(static_cast<int>(boundaryClass));
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

              const auto existing = std::find(
                  signatures.begin(), signatures.end(), signature);
              size_t classId = 0;
              if (existing == signatures.end()) {
                classId = signatures.size();
                signatures.push_back(std::move(signature));
                classes.push_back(AxisPlanClass{fine, coarse});
              } else {
                classId = static_cast<size_t>(existing - signatures.begin());
              }
              classByFine[static_cast<size_t>(fine)] =
                  static_cast<LO>(classId);
            }
          };

          Teuchos::Array<LO> fineXPlanClassHost(
              static_cast<size_t>(localFineNodes[0]));
          Teuchos::Array<LO> fineYPlanClassHost(
              static_cast<size_t>(localFineNodes[1]));
          std::vector<AxisPlanClass> xPlanClasses;
          std::vector<AxisPlanClass> yPlanClasses;

          const size_t numDofs = static_cast<size_t>(dofsPerNode);
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

          int minAcOffsetX = 0;
          int maxAcOffsetX = 0;
          int minAcOffsetY = 0;
          int maxAcOffsetY = 0;
          for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
            const size_t row = maxAcRepresentativeRows[rowDof];
            const size_t sourceNode = row / numDofs;
            const LO sourceX = static_cast<LO>(
                sourceNode % static_cast<size_t>(localCoarseNodes[0]));
            const LO sourceY = static_cast<LO>(
                sourceNode / static_cast<size_t>(localCoarseNodes[0]));
            for (size_t acEntry = acRowMapHost(row);
                 acEntry < acRowMapHost(row + size_t(1)); ++acEntry) {
              const LO targetNode =
                  static_cast<LO>(acEntriesHost(acEntry) / dofsPerNode);
              const LO targetX = targetNode % localCoarseNodes[0];
              const LO targetY = targetNode / localCoarseNodes[0];
              minAcOffsetX = std::min(
                  minAcOffsetX, static_cast<int>(targetX - sourceX));
              maxAcOffsetX = std::max(
                  maxAcOffsetX, static_cast<int>(targetX - sourceX));
              minAcOffsetY = std::min(
                  minAcOffsetY, static_cast<int>(targetY - sourceY));
              maxAcOffsetY = std::max(
                  maxAcOffsetY, static_cast<int>(targetY - sourceY));
            }
          }
          size_t leftAcBoundaryDepth =
              static_cast<size_t>(std::max(0, -minAcOffsetX));
          size_t rightAcBoundaryDepth =
              static_cast<size_t>(std::max(0, maxAcOffsetX));
          size_t bottomAcBoundaryDepth =
              static_cast<size_t>(std::max(0, -minAcOffsetY));
          size_t topAcBoundaryDepth =
              static_cast<size_t>(std::max(0, maxAcOffsetY));
          bool hasInteriorAcRowX = true;
          bool hasInteriorAcRowY = true;
          for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
            const size_t sourceNode =
                maxAcRepresentativeRows[rowDof] / numDofs;
            const size_t sourceX =
                sourceNode % static_cast<size_t>(localCoarseNodes[0]);
            const size_t sourceY =
                sourceNode / static_cast<size_t>(localCoarseNodes[0]);
            hasInteriorAcRowX =
                hasInteriorAcRowX &&
                sourceX >= leftAcBoundaryDepth &&
                static_cast<size_t>(localCoarseNodes[0]) - sourceX - 1 >=
                    rightAcBoundaryDepth;
            hasInteriorAcRowY =
                hasInteriorAcRowY &&
                sourceY >= bottomAcBoundaryDepth &&
                static_cast<size_t>(localCoarseNodes[1]) - sourceY - 1 >=
                    topAcBoundaryDepth;
          }
          if (!hasInteriorAcRowX) {
            leftAcBoundaryDepth =
                static_cast<size_t>(localCoarseNodes[0]);
            rightAcBoundaryDepth = 0;
          }
          if (!hasInteriorAcRowY) {
            bottomAcBoundaryDepth =
                static_cast<size_t>(localCoarseNodes[1]);
            topAcBoundaryDepth = 0;
          }

          buildAxisPlanClasses(
              localFineNodes[0], localCoarseNodes[0], coarseningRate[0],
              minStencilOffsetX, maxStencilOffsetX, leftAcBoundaryDepth,
              rightAcBoundaryDepth, fineXPlanClassHost, xPlanClasses);
          buildAxisPlanClasses(
              localFineNodes[1], localCoarseNodes[1], coarseningRate[1],
              minStencilOffsetY, maxStencilOffsetY, bottomAcBoundaryDepth,
              topAcBoundaryDepth, fineYPlanClassHost, yPlanClasses);
          numXPlanClasses = xPlanClasses.size();
          numYPlanClasses = yPlanClasses.size();

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
              const LO fineBegin =
                  coarse == 0
                      ? LO(0)
                      : coarseAnchor(
                            coarse - 1, numFine, numCoarse, rate) +
                            (coarseAnchor(coarse, numFine, numCoarse, rate) -
                             coarseAnchor(coarse - 1, numFine, numCoarse,
                                          rate)) /
                                LO(2) +
                            1;
              const LO fineEnd =
                  coarse + 1 == numCoarse
                      ? numFine - 1
                      : coarseAnchor(coarse, numFine, numCoarse, rate) +
                            (coarseAnchor(coarse + 1, numFine, numCoarse,
                                          rate) -
                             coarseAnchor(coarse, numFine, numCoarse, rate)) /
                                LO(2);
              const size_t coarseIndex = static_cast<size_t>(coarse);
              const size_t rightDistance =
                  static_cast<size_t>(numCoarse - coarse - LO(1));
              const size_t boundaryClass =
                  coarseIndex < leftDepth
                      ? coarseIndex
                      : (rightDistance < rightDepth
                             ? leftDepth + rightDepth - rightDistance
                             : leftDepth);

              std::vector<int> signature;
              signature.reserve(
                  static_cast<size_t>(fineEnd - fineBegin + LO(2)));
              signature.push_back(static_cast<int>(boundaryClass));
              for (LO fine = fineBegin; fine <= fineEnd; ++fine)
                signature.push_back(static_cast<int>(
                    fineClassByCoordinate[static_cast<size_t>(fine)]));

              const auto existing = std::find(
                  signatures.begin(), signatures.end(), signature);
              size_t classId = 0;
              if (existing == signatures.end()) {
                classId = signatures.size();
                signatures.push_back(std::move(signature));
                classes.push_back(
                    AxisAggregatePlanClass{coarse, fineBegin, fineEnd});
              } else {
                classId = static_cast<size_t>(existing - signatures.begin());
              }
              classByCoarse[static_cast<size_t>(coarse)] =
                  static_cast<LO>(classId);
            }
          };

          Teuchos::Array<LO> coarseXAggregatePlanClassHost(
              static_cast<size_t>(localCoarseNodes[0]));
          Teuchos::Array<LO> coarseYAggregatePlanClassHost(
              static_cast<size_t>(localCoarseNodes[1]));
          std::vector<AxisAggregatePlanClass> xAggregatePlanClasses;
          std::vector<AxisAggregatePlanClass> yAggregatePlanClasses;
          buildAxisAggregatePlanClasses(
              localFineNodes[0], localCoarseNodes[0], coarseningRate[0],
              leftAcBoundaryDepth, rightAcBoundaryDepth,
              fineXPlanClassHost, coarseXAggregatePlanClassHost,
              xAggregatePlanClasses);
          buildAxisAggregatePlanClasses(
              localFineNodes[1], localCoarseNodes[1], coarseningRate[1],
              bottomAcBoundaryDepth, topAcBoundaryDepth,
              fineYPlanClassHost, coarseYAggregatePlanClassHost,
              yAggregatePlanClasses);
          numXAggregatePlanClasses = xAggregatePlanClasses.size();
          numYAggregatePlanClasses = yAggregatePlanClasses.size();
          size_t maxAggregateFineX = 0;
          size_t maxAggregateFineY = 0;
          for (const auto& aggregateClass : xAggregatePlanClasses)
            maxAggregateFineX = std::max(
                maxAggregateFineX,
                static_cast<size_t>(aggregateClass.fineEnd -
                                    aggregateClass.fineBegin + LO(1)));
          for (const auto& aggregateClass : yAggregatePlanClasses)
            maxAggregateFineY = std::max(
                maxAggregateFineY,
                static_cast<size_t>(aggregateClass.fineEnd -
                                    aggregateClass.fineBegin + LO(1)));
          maxFineNodesPerAggregate = maxAggregateFineX * maxAggregateFineY;

          Teuchos::Array<size_t> detectedEntriesPerRow(
              static_cast<size_t>(dofsPerNode), size_t(0));
          for (size_t entry = 0; entry < numFineStencilEntries; ++entry) {
            const LO rowDof = fineStencil.entries[entry].rowDof;
            ++detectedEntriesPerRow[static_cast<size_t>(rowDof)];
          }
          for (LO rowDof = 0; rowDof < dofsPerNode; ++rowDof)
            maxAEntriesPerRow = std::max(
                maxAEntriesPerRow,
                detectedEntriesPerRow[static_cast<size_t>(rowDof)]);
          for (const size_t rowEntries : maxAcEntriesPerDof)
            maxAcEntriesPerRow =
                std::max(maxAcEntriesPerRow, rowEntries);

          coarseXAggregatePlanClass = lo_view(
              "StructuredRAP: coarse-X aggregate plan class",
              static_cast<size_t>(localCoarseNodes[0]));
          coarseYAggregatePlanClass = lo_view(
              "StructuredRAP: coarse-Y aggregate plan class",
              static_cast<size_t>(localCoarseNodes[1]));
          auto coarseXAggregatePlanClassHostView =
              Kokkos::create_mirror_view(coarseXAggregatePlanClass);
          auto coarseYAggregatePlanClassHostView =
              Kokkos::create_mirror_view(coarseYAggregatePlanClass);
          for (size_t coarseX = 0;
               coarseX < static_cast<size_t>(localCoarseNodes[0]);
               ++coarseX)
            coarseXAggregatePlanClassHostView(coarseX) =
                coarseXAggregatePlanClassHost[coarseX];
          for (size_t coarseY = 0;
               coarseY < static_cast<size_t>(localCoarseNodes[1]);
               ++coarseY)
            coarseYAggregatePlanClassHostView(coarseY) =
                coarseYAggregatePlanClassHost[coarseY];
          Kokkos::deep_copy(executionSpace, coarseXAggregatePlanClass,
                            coarseXAggregatePlanClassHostView);
          Kokkos::deep_copy(executionSpace, coarseYAggregatePlanClass,
                            coarseYAggregatePlanClassHostView);

          const size_t planSize = numXPlanClasses * numYPlanClasses *
                                  numDofs * maxAEntriesPerRow;
          const size_t planRowCount =
              numXPlanClasses * numYPlanClasses * numDofs;
          Teuchos::Array<size_t> plannedAcOffsetsHost(
              planSize, invalidStencilNodeOrdinal);
          Teuchos::Array<size_t> plannedARowLengthsHost(
              planRowCount, size_t(0));

          for (size_t yClass = 0; yClass < numYPlanClasses; ++yClass) {
            const LO fineY = yPlanClasses[yClass].representativeFine;
            const LO coarseY = yPlanClasses[yClass].representativeCoarse;
            for (size_t xClass = 0; xClass < numXPlanClasses; ++xClass) {
              const LO fineX = xPlanClasses[xClass].representativeFine;
              const LO coarseX = xPlanClasses[xClass].representativeCoarse;
              const size_t fineNode =
                  static_cast<size_t>(fineY) *
                      static_cast<size_t>(localFineNodes[0]) +
                  static_cast<size_t>(fineX);
              const size_t coarseNode =
                  static_cast<size_t>(coarseY) *
                      static_cast<size_t>(localCoarseNodes[0]) +
                  static_cast<size_t>(coarseX);

              for (LO rowDof = 0; rowDof < dofsPerNode; ++rowDof) {
                const size_t fineRow =
                    fineNode * numDofs + static_cast<size_t>(rowDof);
                const size_t aRowBegin = aRowMapHost(fineRow);
                const size_t aRowEnd = aRowMapHost(fineRow + size_t(1));
                const size_t acRow =
                    coarseNode * numDofs + static_cast<size_t>(rowDof);
                const size_t acRowBegin = acRowMapHost(acRow);
                const size_t acRowEnd = acRowMapHost(acRow + size_t(1));
                const size_t planRowIndex =
                    (yClass * numXPlanClasses + xClass) * numDofs +
                    static_cast<size_t>(rowDof);
                const size_t planRowBegin =
                    planRowIndex * maxAEntriesPerRow;

                plannedARowLengthsHost[planRowIndex] =
                    aRowEnd - aRowBegin;

                for (size_t entryOrdinal = 0;
                     entryOrdinal < aRowEnd - aRowBegin; ++entryOrdinal) {
                  const LO column = aEntriesHost(aRowBegin + entryOrdinal);
                  const LO columnNode = column / dofsPerNode;
                  const LO columnDof = column % dofsPerNode;
                  const LO columnX = columnNode % localFineNodes[0];
                  const LO columnY = columnNode / localFineNodes[0];
                  const LO targetCoarseX = constantInterpolationCoarse(
                      columnX, localFineNodes[0], coarseningRate[0]);
                  const LO targetCoarseY = constantInterpolationCoarse(
                      columnY, localFineNodes[1], coarseningRate[1]);
                  const size_t targetCoarseNode =
                      static_cast<size_t>(targetCoarseY) *
                          static_cast<size_t>(localCoarseNodes[0]) +
                      static_cast<size_t>(targetCoarseX);
                  const size_t targetColumn =
                      targetCoarseNode * numDofs +
                      static_cast<size_t>(columnDof);
                  for (size_t acEntry = acRowBegin; acEntry < acRowEnd;
                       ++acEntry) {
                    if (static_cast<size_t>(acEntriesHost(acEntry)) ==
                        targetColumn) {
                                            plannedAcOffsetsHost[planRowBegin + entryOrdinal] =
                          acEntry - acRowBegin;
                      break;
                    }
                  }
                }
              }
            }
          }

          struct ReverseContribution {
            size_t fineNodeOffset;
            size_t entryOrdinal;
          };
          const size_t reversePlanRowCount =
              numXAggregatePlanClasses * numYAggregatePlanClasses * numDofs;
          const size_t reverseBucketCount =
              reversePlanRowCount * maxAcEntriesPerRow;
          std::vector<std::vector<ReverseContribution>> reverseBuckets(
              reverseBucketCount);
          for (size_t yClass = 0;
               yClass < numYAggregatePlanClasses; ++yClass) {
            const auto& yAggregate = yAggregatePlanClasses[yClass];
            const size_t fineYBegin =
                static_cast<size_t>(yAggregate.fineBegin);
            const size_t fineYEnd =
                static_cast<size_t>(yAggregate.fineEnd);
            for (size_t xClass = 0;
                 xClass < numXAggregatePlanClasses; ++xClass) {
              const auto& xAggregate = xAggregatePlanClasses[xClass];
              const size_t fineXBegin =
                  static_cast<size_t>(xAggregate.fineBegin);
              const size_t fineXEnd =
                  static_cast<size_t>(xAggregate.fineEnd);
              const size_t aggregateWidth = fineXEnd - fineXBegin + 1;
              for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
                const size_t reverseRow =
                    (yClass * numXAggregatePlanClasses + xClass) *
                        numDofs +
                    rowDof;

                for (size_t fineY = fineYBegin; fineY <= fineYEnd; ++fineY) {
                  const size_t sourceYClass = static_cast<size_t>(
                      fineYPlanClassHost[ fineY ]);
                  for (size_t fineX = fineXBegin; fineX <= fineXEnd;
                       ++fineX) {
                    const size_t sourceXClass = static_cast<size_t>(
                        fineXPlanClassHost[ fineX ]);
                    const size_t sourcePlanRow =
                        (sourceYClass * numXPlanClasses + sourceXClass) *
                            numDofs +
                        rowDof;
                    const size_t localFineNodeOffset =
                        (fineY - fineYBegin) * aggregateWidth +
                        fineX - fineXBegin;
                    const size_t sourcePlanBegin =
                        sourcePlanRow * maxAEntriesPerRow;
                    const size_t sourceRowLength =
                        plannedARowLengthsHost[sourcePlanRow];
                    for (size_t entryOrdinal = 0;
                         entryOrdinal < sourceRowLength; ++entryOrdinal) {
                      const size_t acOffset = plannedAcOffsetsHost[
                          sourcePlanBegin + entryOrdinal];
                      if (acOffset == invalidStencilNodeOrdinal)
                        continue;
                      const size_t bucket =
                          reverseRow * maxAcEntriesPerRow + acOffset;
                      reverseBuckets[bucket].push_back(
                          ReverseContribution{localFineNodeOffset,
                                              entryOrdinal});
                    }
                  }
                }
              }
            }
          }

          size_t numReverseContributions = 0;
          for (const auto& bucket : reverseBuckets)
            numReverseContributions += bucket.size();
          reverseContributionBegins = size_view(
              Kokkos::ViewAllocateWithoutInitializing(
                  "StructuredRAP: reverse contribution begins"),
              reverseBucketCount);
          reverseContributionCounts = size_view(
              Kokkos::ViewAllocateWithoutInitializing(
                  "StructuredRAP: reverse contribution counts"),
              reverseBucketCount);
          reverseFineNodeOffsets = size_view(
              Kokkos::ViewAllocateWithoutInitializing(
                  "StructuredRAP: reverse fine-node offsets"),
              numReverseContributions);
          reverseAEntryOrdinals = size_view(
              Kokkos::ViewAllocateWithoutInitializing(
                  "StructuredRAP: reverse A entry ordinals"),
              numReverseContributions);
          auto reverseContributionBeginsHost =
              Kokkos::create_mirror_view(reverseContributionBegins);
          auto reverseContributionCountsHost =
              Kokkos::create_mirror_view(reverseContributionCounts);
          auto reverseFineNodeOffsetsHost =
              Kokkos::create_mirror_view(reverseFineNodeOffsets);
          auto reverseAEntryOrdinalsHost =
              Kokkos::create_mirror_view(reverseAEntryOrdinals);
          size_t reverseContribution = 0;
          for (size_t bucket = 0; bucket < reverseBucketCount; ++bucket) {
            reverseContributionBeginsHost(bucket) = reverseContribution;
            reverseContributionCountsHost(bucket) =
                reverseBuckets[bucket].size();
            for (const auto& source : reverseBuckets[bucket]) {
              reverseFineNodeOffsetsHost(reverseContribution) =
                  source.fineNodeOffset;
              reverseAEntryOrdinalsHost(reverseContribution) =
                  source.entryOrdinal;
              ++reverseContribution;
            }
          }
          Kokkos::deep_copy(executionSpace, reverseContributionBegins,
                            reverseContributionBeginsHost);
          Kokkos::deep_copy(executionSpace, reverseContributionCounts,
                            reverseContributionCountsHost);
          Kokkos::deep_copy(executionSpace, reverseFineNodeOffsets,
                            reverseFineNodeOffsetsHost);
          Kokkos::deep_copy(executionSpace, reverseAEntryOrdinals,
                            reverseAEntryOrdinalsHost);

          executionSpace.fence("StructuredRAP: numeric entry plan prepared");
        }

        const size_t numCoarseNodeTiles =
            (numCoarseNodes + coarseNodesPerTile - size_t(1)) /
            coarseNodesPerTile;
        fineARowBegins = size_view(
            Kokkos::ViewAllocateWithoutInitializing(
                "StructuredRAP: tile-local fine A row begins"),
            numCoarseNodeTiles * maxFineNodesPerAggregate * numScalarDofs);

        {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(
                  timerPrefix +
                  "numeric (Elasticity2D constant stencil)"));
          range_policy policy(executionSpace, 0, numCoarseNodeTiles);
          Kokkos::parallel_for(
              "StructuredRAP: Elasticity2D constant stencil P^T A P",
              policy,
              KOKKOS_LAMBDA(const size_t tileIndex) {
                const size_t firstCoarseNode =
                    tileIndex * coarseNodesPerTile;
                const size_t lastCoarseNode = Kokkos::min(
                    numCoarseNodes, firstCoarseNode + coarseNodesPerTile);
                const size_t tileFineRowsBegin =
                    tileIndex * maxFineNodesPerAggregate * numScalarDofs;
                for (size_t coarseNodeIndex = firstCoarseNode;
                     coarseNodeIndex < lastCoarseNode;
                     ++coarseNodeIndex) {
                const LO coarseNode = static_cast<LO>(coarseNodeIndex);
                const LO coarseX = coarseNode % localCoarseNodes[0];
                const LO coarseY = coarseNode / localCoarseNodes[0];
                const size_t firstCoarseRow =
                    coarseNodeIndex * numScalarDofs;

                const LO fineXBegin =
                    coarseX == 0
                        ? LO(0)
                        : coarseAnchor(
                              coarseX - 1, localFineNodes[0],
                              localCoarseNodes[0], coarseningRate[0]) +
                              (coarseAnchor(
                                   coarseX, localFineNodes[0],
                                   localCoarseNodes[0], coarseningRate[0]) -
                               coarseAnchor(
                                   coarseX - 1, localFineNodes[0],
                                   localCoarseNodes[0], coarseningRate[0])) /
                                  LO(2) +
                              1;
                const LO fineXEnd =
                    coarseX + 1 == localCoarseNodes[0]
                        ? localFineNodes[0] - 1
                        : coarseAnchor(
                              coarseX, localFineNodes[0],
                              localCoarseNodes[0], coarseningRate[0]) +
                              (coarseAnchor(
                                   coarseX + 1, localFineNodes[0],
                                   localCoarseNodes[0], coarseningRate[0]) -
                               coarseAnchor(
                                   coarseX, localFineNodes[0],
                                   localCoarseNodes[0], coarseningRate[0])) /
                                  LO(2);
                const LO fineYBegin =
                    coarseY == 0
                        ? LO(0)
                        : coarseAnchor(
                              coarseY - 1, localFineNodes[1],
                              localCoarseNodes[1], coarseningRate[1]) +
                              (coarseAnchor(
                                   coarseY, localFineNodes[1],
                                   localCoarseNodes[1], coarseningRate[1]) -
                               coarseAnchor(
                                   coarseY - 1, localFineNodes[1],
                                   localCoarseNodes[1], coarseningRate[1])) /
                                  LO(2) +
                              1;
                const LO fineYEnd =
                    coarseY + 1 == localCoarseNodes[1]
                        ? localFineNodes[1] - 1
                        : coarseAnchor(
                              coarseY, localFineNodes[1],
                              localCoarseNodes[1], coarseningRate[1]) +
                              (coarseAnchor(
                                   coarseY + 1, localFineNodes[1],
                                   localCoarseNodes[1], coarseningRate[1]) -
                               coarseAnchor(
                                   coarseY, localFineNodes[1],
                                   localCoarseNodes[1], coarseningRate[1])) /
                                  LO(2);

                const size_t aggregateWidth =
                    static_cast<size_t>(fineXEnd - fineXBegin + LO(1));
                const size_t aggregateXClass =
                    static_cast<size_t>(coarseXAggregatePlanClass(
                        static_cast<size_t>(coarseX)));
                const size_t aggregateYClass =
                    static_cast<size_t>(coarseYAggregatePlanClass(
                        static_cast<size_t>(coarseY)));
                const size_t reversePlanNodeBegin =
                    (aggregateYClass * numXAggregatePlanClasses +
                     aggregateXClass) * numScalarDofs;

                for (LO fineY = fineYBegin; fineY <= fineYEnd; ++fineY) {
                  for (LO fineX = fineXBegin; fineX <= fineXEnd; ++fineX) {
                    const size_t fineNode =
                        static_cast<size_t>(fineY) *
                            static_cast<size_t>(localFineNodes[0]) +
                        static_cast<size_t>(fineX);
                    const size_t localFineNodeOffset =
                        static_cast<size_t>(fineY - fineYBegin) *
                            aggregateWidth +
                        static_cast<size_t>(fineX - fineXBegin);
                    for (size_t rowDof = 0; rowDof < numScalarDofs;
                         ++rowDof) {
                      const size_t localFineRow =
                          localFineNodeOffset * numScalarDofs + rowDof;
                      const size_t aRow =
                          fineNode * numScalarDofs + rowDof;
                      fineARowBegins(tileFineRowsBegin + localFineRow) =
                          localA.graph.row_map(aRow);
                    }
                  }
                }

                for (size_t rowDof = 0; rowDof < numScalarDofs; ++rowDof) {
                      const size_t acRow = firstCoarseRow + rowDof;
                      const size_t acRowBegin =
                          localAc.graph.row_map(acRow);
                      const size_t acRowEnd =
                          localAc.graph.row_map(acRow + size_t(1));
                      const size_t reversePlanRow =
                          reversePlanNodeBegin + rowDof;
                      for (size_t acOffset = 0;
                           acOffset < acRowEnd - acRowBegin; ++acOffset) {
                        const size_t bucket =
                            reversePlanRow * maxAcEntriesPerRow + acOffset;
                        const size_t contributionBegin =
                            reverseContributionBegins(bucket);
                        const size_t contributionEnd =
                            contributionBegin +
                            reverseContributionCounts(bucket);
                        SC sum = zero;
                        for (size_t contribution = contributionBegin;
                             contribution < contributionEnd; ++contribution) {
                          const size_t localFineNodeOffset =
                              reverseFineNodeOffsets(contribution);
                          const size_t entryOrdinal =
                              reverseAEntryOrdinals(contribution);
                          const size_t localFineRow =
                              localFineNodeOffset * numScalarDofs + rowDof;
                          sum += localA.values(
                              fineARowBegins(tileFineRowsBegin +
                                             localFineRow) +
                              entryOrdinal);
                        }
                        localAc.values(acRowBegin + acOffset) = sum;
                      }
                    }
                }
              });
          executionSpace.fence(
              "StructuredRAP: Elasticity2D stencil product complete");
        }

        if (!Ac.isFillComplete()) {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(timerPrefix +
                                                 "fillComplete"));
          Ac.fillComplete(P.getDomainMap(), P.getDomainMap(), params);
        }
        return;
      }

    throw std::runtime_error(
        "StructuredRAPKernel supports only single-rank 2D order-zero structured RAP.");
  }
};

}  // namespace Details
}  // namespace MueLu

#endif  // MUELU_STRUCTUREDRAPKERNEL_HPP
