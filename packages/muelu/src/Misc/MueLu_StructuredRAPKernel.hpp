// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef MUELU_STRUCTUREDRAPKERNEL_HPP
#define MUELU_STRUCTUREDRAPKERNEL_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_Import.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixFactory.hpp>

namespace MueLu {
namespace Details {

/**
 * Fused numeric kernel for Ac = P^T A P on a supplied coarse graph.
 *
 * The caller owns construction and validation of the structured Ac graph.  The
 * numeric path derives order-zero and order-one interpolation weights from the
 * structured fine-grid indices and caches them with the P graph. Rows whose
 * interpolation geometry cannot be resolved locally retain an exact sparse-P
 * fallback. On multiple ranks, contributions from locally owned fine rows are
 * accumulated into overlapping coarse rows and exported to the owning ranks.
 */
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class StructuredRAPKernel {
 public:
  using Matrix        = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using Graph         = Xpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;
  using Import        = Xpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
  using MatrixFactory = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using GraphFactory  = Xpetra::CrsGraphFactory<LocalOrdinal, GlobalOrdinal, Node>;

 private:
  using device_type    = typename Node::device_type;
  using size_view      = Kokkos::View<size_t*, device_type>;
  using lo_view        = Kokkos::View<LocalOrdinal*, device_type>;
  using const_size_view = Kokkos::View<const size_t*, device_type>;
  using const_lo_view   = Kokkos::View<const LocalOrdinal*, device_type>;
  using scalar_view    = Kokkos::View<Scalar*, device_type>;
  using int_view       = Kokkos::View<int*, device_type>;

  struct GraphCache {
    Teuchos::RCP<const Graph> aGraph;
    Teuchos::RCP<const Graph> pGraph;
    Teuchos::RCP<const Graph> acGraph;
    Teuchos::RCP<Matrix> importedP;
    Teuchos::RCP<Matrix> overlapAc;
    size_view transposeRowPtr;
    lo_view transposeFineRows;
    size_view transposePEntries;
    scalar_view localGeometricPValues;
    int_view localGeometricRows;
    const_size_view ghostedGeometricRowPtr;
    const_lo_view ghostedGeometricColumns;
    scalar_view ghostedGeometricPValues;
    int_view ghostedGeometricRows;
    int interpolationOrder = -1;
    int numDimensions      = 0;
    Kokkos::Array<LocalOrdinal, 3> localFineNodes{};
    lo_view ghostedPToAcCol;
    bool pToAcColumnMapIsIdentity = false;
    size_t maxAcRowEntries        = 0;
  };

  static constexpr const char* cacheKey() {
    return "structured rap: graph cache";
  }

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

  KOKKOS_INLINE_FUNCTION
  static int linearInterpolationSupport(const LocalOrdinal fine,
                                        const LocalOrdinal numFine,
                                        const LocalOrdinal numCoarse,
                                        const LocalOrdinal rate,
                                        LocalOrdinal* coarse,
                                        Scalar* weight) {
    if (numCoarse == 1) {
      coarse[0] = 0;
      weight[0] = Scalar(1);
      return 1;
    }
    if (fine == numFine - 1) {
      coarse[0] = numCoarse - 1;
      weight[0] = Scalar(1);
      return 1;
    }

    LocalOrdinal low       = fine / rate;
    const LocalOrdinal lowAnchor = low * rate;
    if (fine == lowAnchor) {
      coarse[0] = low;
      weight[0] = Scalar(1);
      return 1;
    }
    if (low + 1 >= numCoarse)
      low = numCoarse - 2;
    const LocalOrdinal high       = low + 1;
    const LocalOrdinal left       = coarseAnchor(low, numFine, numCoarse, rate);
    const LocalOrdinal right      = coarseAnchor(high, numFine, numCoarse, rate);
    const Scalar denominator      = static_cast<Scalar>(right - left);
    coarse[0] = low;
    coarse[1] = high;
    weight[0] = static_cast<Scalar>(right - fine) / denominator;
    weight[1] = static_cast<Scalar>(fine - left) / denominator;
    return 2;
  }

  KOKKOS_INLINE_FUNCTION
  static Scalar linearInterpolationWeight(const LocalOrdinal fine,
                                          const LocalOrdinal coarse,
                                          const LocalOrdinal numFine,
                                          const LocalOrdinal numCoarse,
                                          const LocalOrdinal rate) {
    LocalOrdinal support[2];
    Scalar weights[2];
    const int count = linearInterpolationSupport(
        fine, numFine, numCoarse, rate, support, weights);
    for (int entry = 0; entry < count; ++entry)
      if (support[entry] == coarse)
        return weights[entry];
    return Scalar(0);
  }

 public:
  static void Compute(const Matrix& A,
                      const Matrix& P,
                      Matrix& Ac,
                      const std::string& matrixType,
                      const int interpolationOrder,
                      const int numDimensions,
                      const Teuchos::Array<LocalOrdinal>& lFineNodesPerDim,
                      const Teuchos::Array<LocalOrdinal>& lCoarseNodesPerDim,
                      const Teuchos::Array<int>& structuredCoarseningRate,
                      const Teuchos::RCP<Teuchos::ParameterList>& params = Teuchos::null) {
    using LO              = LocalOrdinal;
    using GO              = GlobalOrdinal;
    using SC              = Scalar;
    using execution_space = typename device_type::execution_space;
    using range_policy    = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_t>>;
    using team_policy     = Kokkos::TeamPolicy<execution_space>;
    using member_type     = typename team_policy::member_type;
    static constexpr bool executionSpaceIsHost =
        std::is_same<typename device_type::memory_space, Kokkos::HostSpace>::value;
    using scratch_view    = Kokkos::View<SC*, typename execution_space::scratch_memory_space,
                                         Kokkos::MemoryUnmanaged>;

    const std::string prefix      = "MueLu::Details::StructuredRAPKernel::Compute: ";
    const std::string timerPrefix = "MueLu: StructuredRAPKernel: ";

    TEUCHOS_TEST_FOR_EXCEPTION(interpolationOrder < 0 || interpolationOrder > 1,
                               std::runtime_error,
                               prefix << "only interpolation orders zero and one are supported.");
    TEUCHOS_TEST_FOR_EXCEPTION(numDimensions < 1 || numDimensions > 3,
                               std::runtime_error,
                               prefix << "the number of dimensions must be between one and three.");
    TEUCHOS_TEST_FOR_EXCEPTION(lFineNodesPerDim.size() < numDimensions,
                               std::runtime_error,
                               prefix << "insufficient fine-grid dimensions were supplied.");
    Kokkos::Array<LO, 3> localFineNodes{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    Kokkos::Array<LO, 3> localCoarseNodes{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    Kokkos::Array<LO, 3> coarseningRate{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    for (int dim = 0; dim < numDimensions; ++dim) {
      TEUCHOS_TEST_FOR_EXCEPTION(lFineNodesPerDim[dim] <= 0, std::runtime_error,
                                 prefix << "fine-grid dimensions must be positive.");
      localFineNodes[dim] = lFineNodesPerDim[dim];
    }
    const bool hasStructuredCoarseningRate = !structuredCoarseningRate.empty();
    if (hasStructuredCoarseningRate) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          lCoarseNodesPerDim.size() < numDimensions,
          std::runtime_error,
          prefix << "insufficient coarse-grid dimensions were supplied.");
      TEUCHOS_TEST_FOR_EXCEPTION(
          structuredCoarseningRate.size() != 1 &&
              structuredCoarseningRate.size() < numDimensions,
          std::runtime_error,
          prefix << "the structured coarsening rate must contain one value or "
                    "one value per active dimension.");
      for (int dim = 0; dim < numDimensions; ++dim) {
        const int rate = structuredCoarseningRate.size() == 1
                             ? structuredCoarseningRate[0]
                             : structuredCoarseningRate[dim];
        TEUCHOS_TEST_FOR_EXCEPTION(rate <= 0 || lCoarseNodesPerDim[dim] <= 0,
                                   std::runtime_error,
                                   prefix << "coarsening rates and coarse-grid dimensions must be positive.");
        localCoarseNodes[dim] = lCoarseNodesPerDim[dim];
        coarseningRate[dim]   = static_cast<LO>(rate);
      }
    }

    TEUCHOS_TEST_FOR_EXCEPTION(A.getRowMap()->lib() != Xpetra::UseTpetra,
                               std::runtime_error,
                               prefix << "only the Tpetra backend is supported.");
    TEUCHOS_TEST_FOR_EXCEPTION(!A.isFillComplete() || !P.isFillComplete(),
                               std::runtime_error,
                               prefix << "A and P must be fill complete.");
    TEUCHOS_TEST_FOR_EXCEPTION(!A.hasCrsGraph() || !P.hasCrsGraph() || !Ac.hasCrsGraph(),
                               std::runtime_error,
                               prefix << "A, P, and Ac must expose CrsGraph objects.");
    TEUCHOS_TEST_FOR_EXCEPTION(!A.getRowMap()->isSameAs(*P.getRowMap()),
                               std::runtime_error,
                               prefix << "the row maps of A and P must match.");
    TEUCHOS_TEST_FOR_EXCEPTION(!P.getDomainMap()->isSameAs(*Ac.getRowMap()),
                               std::runtime_error,
                               prefix << "the domain map of P must match the row map of Ac.");

    execution_space executionSpace;
    if constexpr (executionSpaceIsHost) {
      bool structuredDimensionsMatch = hasStructuredCoarseningRate;
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
          static_cast<size_t>(localFineNodes[1]) * size_t(2);
      const size_t expectedCoarseRows =
          static_cast<size_t>(localCoarseNodes[0]) *
          static_cast<size_t>(localCoarseNodes[1]) * size_t(2);
      const bool useElasticity2DStencil =
          matrixType == "Elasticity2D" &&
          interpolationOrder == 0 &&
          numDimensions == 2 &&
          structuredDimensionsMatch &&
          A.GetFixedBlockSize() == 2 &&
          localCoarseNodes[0] >= LO(3) &&
          A.getRowMap()->getComm()->getSize() == 1 &&
          A.getColMap()->isSameAs(*A.getRowMap()) &&
          Ac.getColMap()->isSameAs(*Ac.getRowMap()) &&
          A.getRowMap()->getLocalNumElements() == expectedFineRows &&
          Ac.getRowMap()->getLocalNumElements() == expectedCoarseRows;

      if (useElasticity2DStencil) {
        const SC zero = Teuchos::ScalarTraits<SC>::zero();
        const auto localA = A.getLocalMatrixDevice();
        auto localAc      = Ac.getLocalMatrixDevice();
        const size_t numFineNodes =
            static_cast<size_t>(localFineNodes[0]) *
            static_cast<size_t>(localFineNodes[1]);
        const size_t numCoarseNodes =
            static_cast<size_t>(localCoarseNodes[0]) *
            static_cast<size_t>(localCoarseNodes[1]);
        const size_t numWorkers =
            static_cast<size_t>(executionSpace.concurrency());

        // Build the order-zero interpolation map once. The numeric kernel can
        // then map every A column to its single coarse-grid owner with one
        // contiguous lookup instead of repeating coordinate divisions and
        // boundary logic for every matrix entry.
        lo_view fineNodeToCoarseNode(
            Kokkos::ViewAllocateWithoutInitializing(
                "StructuredRAP: constant fine-to-coarse map"),
            numFineNodes);
        {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(
                  timerPrefix + "build constant interpolation map"));
          Kokkos::parallel_for(
              "StructuredRAP: build constant fine-to-coarse map",
              range_policy(executionSpace, 0, numFineNodes),
              KOKKOS_LAMBDA(const size_t fineNodeIndex) {
                const LO fineNode = static_cast<LO>(fineNodeIndex);
                const LO fineX    = fineNode % localFineNodes[0];
                const LO fineY    = fineNode / localFineNodes[0];
                const LO coarseX  = constantInterpolationCoarse(
                    fineX, localFineNodes[0], coarseningRate[0]);
                const LO coarseY  = constantInterpolationCoarse(
                    fineY, localFineNodes[1], coarseningRate[1]);
                fineNodeToCoarseNode(fineNodeIndex) =
                    coarseY * localCoarseNodes[0] + coarseX;
              });
          executionSpace.fence(
              "StructuredRAP: constant interpolation map complete");
        }

        Kokkos::View<int, device_type> missingStencilEntry(
            "StructuredRAP: missing stencil graph entry");
        Kokkos::deep_copy(executionSpace, missingStencilEntry, 0);

        {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(
                  timerPrefix +
                  "numeric (Elasticity2D constant stencil)"));
          Kokkos::parallel_for(
              "StructuredRAP: Elasticity2D constant stencil P^T A P",
              range_policy(executionSpace, 0, numWorkers),
              KOKKOS_LAMBDA(const size_t worker) {
                const size_t firstCoarseNode =
                    worker * numCoarseNodes / numWorkers;
                const size_t lastCoarseNode =
                    (worker + 1) * numCoarseNodes / numWorkers;

                for (size_t coarseNodeIndex = firstCoarseNode;
                     coarseNodeIndex < lastCoarseNode;
                     ++coarseNodeIndex) {
                  const LO coarseNode =
                      static_cast<LO>(coarseNodeIndex);
                  const size_t firstCoarseRow = coarseNodeIndex * size_t(2);
                  const size_t acBegin0 =
                      localAc.graph.row_map(firstCoarseRow);
                  const size_t acEnd0 =
                      localAc.graph.row_map(firstCoarseRow + 1);
                  const size_t acBegin1 = acEnd0;
                  const size_t acEnd1 =
                      localAc.graph.row_map(firstCoarseRow + 2);
                  for (size_t acEntry = acBegin0;
                       acEntry < acEnd1; ++acEntry) {
                    localAc.values(acEntry) = zero;
                  }

                  const LO coarseX =
                      coarseNode % localCoarseNodes[0];
                  const LO coarseY =
                      coarseNode / localCoarseNodes[0];
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

                  const LO firstStencilX = coarseX == 0 ? LO(0) : coarseX - 1;
                  const LO firstStencilY = coarseY == 0 ? LO(0) : coarseY - 1;
                  const LO lastStencilX =
                      coarseX + 1 < localCoarseNodes[0] ? coarseX + 1 : coarseX;
                  const LO lastStencilY =
                      coarseY + 1 < localCoarseNodes[1] ? coarseY + 1 : coarseY;
                  const LO stencilWidth = lastStencilX - firstStencilX + 1;
                  const LO stencilHeight = lastStencilY - firstStencilY + 1;
                  const size_t expectedRowLength =
                      static_cast<size_t>(stencilWidth) *
                      static_cast<size_t>(stencilHeight) * size_t(2);
                  bool validStencilRows =
                      acEnd0 - acBegin0 == expectedRowLength &&
                      acEnd1 - acBegin1 == expectedRowLength;
                  for (LO stencilY = firstStencilY;
                       stencilY <= lastStencilY && validStencilRows;
                       ++stencilY) {
                    for (LO stencilX = firstStencilX;
                         stencilX <= lastStencilX && validStencilRows;
                         ++stencilX) {
                      const LO stencilNode =
                          stencilY * localCoarseNodes[0] + stencilX;
                      const size_t nodeOffset = static_cast<size_t>(
                          (stencilY - firstStencilY) * stencilWidth +
                          (stencilX - firstStencilX)) * size_t(2);
                      for (LO columnDof = 0; columnDof < LO(2); ++columnDof) {
                        const LO expectedColumn = stencilNode * LO(2) + columnDof;
                        const size_t entryOffset =
                            nodeOffset + static_cast<size_t>(columnDof);
                        validStencilRows = validStencilRows &&
                            localAc.graph.entries(acBegin0 + entryOffset) == expectedColumn &&
                            localAc.graph.entries(acBegin1 + entryOffset) == expectedColumn;
                      }
                    }
                  }
                  if (!validStencilRows) {
                    Kokkos::atomic_exchange(&missingStencilEntry(), 1);
                    continue;
                  }

                  for (LO fineY = fineYBegin; fineY <= fineYEnd;
                       ++fineY) {
                    for (LO fineX = fineXBegin; fineX <= fineXEnd;
                         ++fineX) {
                      const LO fineNode =
                          fineY * localFineNodes[0] + fineX;
                      for (LO coarseDof = 0; coarseDof < LO(2);
                           ++coarseDof) {
                        const LO fineRow =
                            fineNode * LO(2) + coarseDof;
                        const auto aRow = localA.rowConst(fineRow);
                        const size_t destinationBegin =
                            coarseDof == 0 ? acBegin0 : acBegin1;

                        for (LO aEntry = 0; aEntry < aRow.length;
                             ++aEntry) {
                          const LO fineColumn = aRow.colidx(aEntry);
                          const LO columnDof  = fineColumn % LO(2);
                          const LO columnNode = fineColumn / LO(2);
                          const LO targetNode =
                              fineNodeToCoarseNode(
                                  static_cast<size_t>(columnNode));

                          // The sorted full 3x3 coarse stencil gives the exact
                          // value-array offset. Classifying the nodal delta
                          // avoids coordinate division and sparse lookup.
                          const LO nodeDelta = targetNode - coarseNode;
                          LO targetDx        = nodeDelta;
                          LO targetDy        = 0;
                          if (nodeDelta < -1) {
                            targetDy = -1;
                            targetDx += localCoarseNodes[0];
                          } else if (nodeDelta > 1) {
                            targetDy = 1;
                            targetDx -= localCoarseNodes[0];
                          }
                          const LO targetX = coarseX + targetDx;
                          const LO targetY = coarseY + targetDy;
                          const bool targetInStencil =
                              targetDx >= -1 && targetDx <= 1 &&
                              targetDy >= -1 && targetDy <= 1 &&
                              targetX >= firstStencilX &&
                              targetX <= lastStencilX &&
                              targetY >= firstStencilY &&
                              targetY <= lastStencilY;
                          if (!targetInStencil) {
                            Kokkos::atomic_exchange(
                                &missingStencilEntry(), 1);
                            continue;
                          }
                          const size_t entryOffset = static_cast<size_t>(
                              (targetY - firstStencilY) * stencilWidth +
                              (targetX - firstStencilX)) * size_t(2) +
                              static_cast<size_t>(columnDof);
                          // Piecewise-constant P and R=P^T both contribute
                          // unit weights; no transfer entry is accessed.
                          localAc.values(destinationBegin + entryOffset) +=
                              aRow.value(aEntry);
                        }
                      }
                    }
                  }
                }
              });
          executionSpace.fence(
              "StructuredRAP: Elasticity2D stencil product complete");
        }

        int missingStencilEntryHost = 0;
        Kokkos::deep_copy(executionSpace, missingStencilEntryHost,
                          missingStencilEntry);
        TEUCHOS_TEST_FOR_EXCEPTION(
            missingStencilEntryHost != 0, std::runtime_error,
            prefix << "the supplied Ac graph does not contain every entry "
                      "generated by the Elasticity2D stencil kernel.");

        if (!Ac.isFillComplete()) {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(timerPrefix +
                                                 "fillComplete"));
          Ac.fillComplete(P.getDomainMap(), P.getDomainMap(), params);
        }
        return;
      }
    }

    const Teuchos::RCP<const Graph> aGraph  = A.getCrsGraph();
    const Teuchos::RCP<const Graph> pGraph  = P.getCrsGraph();
    const Teuchos::RCP<const Graph> acGraph = Ac.getCrsGraph();
    Teuchos::RCP<GraphCache> cache;
    if (!params.is_null() && params->isType<Teuchos::RCP<GraphCache>>(cacheKey())) {
      cache = params->get<Teuchos::RCP<GraphCache>>(cacheKey());
      params->remove(cacheKey());
    }

    bool cacheValid = !cache.is_null() &&
                      cache->aGraph.getRawPtr() == aGraph.getRawPtr() &&
                      cache->pGraph.getRawPtr() == pGraph.getRawPtr() &&
                      cache->acGraph.getRawPtr() == acGraph.getRawPtr() &&
                      cache->interpolationOrder == interpolationOrder &&
                      cache->numDimensions == numDimensions;
    for (int dim = 0; dim < 3 && cacheValid; ++dim)
      cacheValid = cache->localFineNodes[dim] == localFineNodes[dim];
    if (!cacheValid) {
      cache          = Teuchos::rcp(new GraphCache);
      cache->aGraph  = aGraph;
      cache->pGraph  = pGraph;
      cache->acGraph = acGraph;
      cache->interpolationOrder = interpolationOrder;
      cache->numDimensions      = numDimensions;
      cache->localFineNodes     = localFineNodes;
    }

    const Teuchos::RCP<const Matrix> Pptr = Teuchos::rcpFromRef(P);

    // A's column map identifies every P row needed by the local A rows.  Import
    // those rows before the device kernel so all numeric traversal is local.
    const Teuchos::RCP<const Import> fineRowImporter = A.getCrsGraph()->getImporter();
    Teuchos::RCP<const Matrix> ghostedP              = Pptr;
    Teuchos::RCP<Teuchos::TimeMonitor> importTimer = Teuchos::rcp(
        new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "import P")));
    if (!fineRowImporter.is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!fineRowImporter->getSourceMap()->isSameAs(*P.getRowMap()),
                                 std::runtime_error,
                                 prefix << "A's importer source map does not match P's row map.");
      if (cacheValid) {
        TEUCHOS_TEST_FOR_EXCEPTION(cache->importedP.is_null(), std::runtime_error,
                                   prefix << "cached imported P is null.");
        cache->importedP->setAllToScalar(Teuchos::ScalarTraits<SC>::zero());
        cache->importedP->doImport(P, *fineRowImporter, Xpetra::ADD);
      } else {
        cache->importedP = MatrixFactory::Build(Pptr, *fineRowImporter);
      }
      ghostedP = cache->importedP;
      TEUCHOS_TEST_FOR_EXCEPTION(ghostedP.is_null(),
                                 std::runtime_error,
                                 prefix << "failed to import the P rows required by A's column map.");
    }
    importTimer = Teuchos::null;

    // A local fine row can contribute to coarse rows owned by another rank.
    // Import the supplied Ac graph onto P's column map, compute those overlap
    // rows, and sum them back to the owning Ac rows after the device kernel.
    const Teuchos::RCP<const Import> coarseRowImporter = P.getCrsGraph()->getImporter();
    Teuchos::RCP<Matrix> overlapAc                     = Teuchos::rcpFromRef(Ac);
    Teuchos::RCP<Teuchos::TimeMonitor> overlapTimer = Teuchos::rcp(
        new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "build overlap Ac")));
    if (!coarseRowImporter.is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!coarseRowImporter->getSourceMap()->isSameAs(*Ac.getRowMap()),
                                 std::runtime_error,
                                 prefix << "P's importer source map does not match Ac's row map.");
      if (cacheValid) {
        overlapAc = cache->overlapAc;
      } else {
        const auto overlapGraph = GraphFactory::Build(acGraph, *coarseRowImporter,
                                                      Ac.getDomainMap(), Ac.getRangeMap());
        cache->overlapAc         = MatrixFactory::Build(overlapGraph.getConst());
        overlapAc                = cache->overlapAc;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(overlapAc.is_null(), std::runtime_error,
                                 prefix << "cached overlapping Ac is null.");
    }
    overlapTimer = Teuchos::null;

    const SC zero = Teuchos::ScalarTraits<SC>::zero();
    const SC one  = Teuchos::ScalarTraits<SC>::one();
    Teuchos::RCP<Teuchos::TimeMonitor> initializeTimer = Teuchos::rcp(
        new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "initialize values")));
    if constexpr (executionSpaceIsHost) {
      // The CPU kernel initializes overlapAc while it constructs its dense
      // row lookup. Only the export target needs a separate zeroing pass.
      if (overlapAc.getRawPtr() != &Ac)
        Ac.setAllToScalar(zero);
    } else {
      Ac.setAllToScalar(zero);
      if (overlapAc.getRawPtr() != &Ac)
        overlapAc->setAllToScalar(zero);
    }
    Kokkos::fence("StructuredRAP: value initialization complete");
    initializeTimer = Teuchos::null;

    const auto localA         = A.getLocalMatrixDevice();
    const auto localP         = P.getLocalMatrixDevice();
    const auto localGhostedP  = ghostedP->getLocalMatrixDevice();
    auto localOverlapAc       = overlapAc->getLocalMatrixDevice();
    const size_t numFineRows  = localA.numRows();
    const size_t numCoarseRows = localOverlapAc.numRows();
    const LO dofsPerNode       = Teuchos::as<LO>(A.GetFixedBlockSize());
    const size_t numFineNodes  = static_cast<size_t>(localFineNodes[0]) *
                                 static_cast<size_t>(localFineNodes[1]) *
                                 static_cast<size_t>(localFineNodes[2]);
    TEUCHOS_TEST_FOR_EXCEPTION(dofsPerNode <= 0, std::runtime_error,
                               prefix << "A must have a positive fixed block size.");
    TEUCHOS_TEST_FOR_EXCEPTION(
        numFineNodes * static_cast<size_t>(dofsPerNode) != numFineRows,
        std::runtime_error,
        prefix << "the structured fine-grid dimensions and A's fixed block size "
                  "do not match A's local row count.");

    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(localP.numRows()) != numFineRows,
                               std::runtime_error,
                               prefix << "A and P have different local row counts.");
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(localGhostedP.numRows()) != A.getColMap()->getLocalNumElements(),
                               std::runtime_error,
                               prefix << "the imported P row map does not match A's column map.");
    TEUCHOS_TEST_FOR_EXCEPTION(numCoarseRows != P.getColMap()->getLocalNumElements(),
                               std::runtime_error,
                               prefix << "the overlap Ac row map does not match P's column map.");

    size_view transposeRowPtr;
    lo_view transposeFineRows;
    size_view transposePEntries;
    lo_view ghostedPToAcCol;
    bool pToAcColumnMapIsIdentity = false;
    const LO invalidLO          = Teuchos::OrdinalTraits<LO>::invalid();

    Kokkos::View<int, device_type> missingGraphEntry("StructuredRAP: missing graph entry");
    Kokkos::deep_copy(executionSpace, missingGraphEntry, 0);

    if (!cacheValid) {
      {
        Teuchos::TimeMonitor timer(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "build transpose adjacency"));
        // Form a lightweight local transpose adjacency for P.  Values remain in P;
        // only fine-row and P-entry indices are stored here.
        size_view transposeCounts("StructuredRAP: transpose counts", numCoarseRows);
        if constexpr (executionSpaceIsHost) {
          for (size_t row = 0; row < numCoarseRows; ++row)
            transposeCounts(row) = 0;
          for (size_t fineRow = 0; fineRow < numFineRows; ++fineRow) {
            const auto pRow = localP.rowConst(static_cast<LO>(fineRow));
            for (LO entry = 0; entry < pRow.length; ++entry)
              ++transposeCounts(static_cast<size_t>(pRow.colidx(entry)));
          }
        } else {
          Kokkos::deep_copy(executionSpace, transposeCounts, size_t(0));
          Kokkos::parallel_for(
              "StructuredRAP: count transpose entries", range_policy(executionSpace, 0, numFineRows),
              KOKKOS_LAMBDA(const size_t fineRow) {
                const auto pRow = localP.rowConst(static_cast<LO>(fineRow));
                for (LO entry = 0; entry < pRow.length; ++entry)
                  Kokkos::atomic_fetch_add(&transposeCounts(static_cast<size_t>(pRow.colidx(entry))), size_t(1));
              });
        }

        transposeRowPtr = size_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose row pointers"), numCoarseRows + 1);
        if constexpr (executionSpaceIsHost) {
          transposeRowPtr(0) = 0;
          for (size_t row = 0; row < numCoarseRows; ++row)
            transposeRowPtr(row + 1) = transposeRowPtr(row) + transposeCounts(row);
        } else {
          Kokkos::parallel_scan(
              "StructuredRAP: scan transpose rows", range_policy(executionSpace, 0, numCoarseRows + 1),
              KOKKOS_LAMBDA(const size_t row, size_t& update, const bool final) {
                if (final)
                  transposeRowPtr(row) = update;
                if (row < numCoarseRows)
                  update += transposeCounts(row);
              });
        }

        const size_t localPEntries = localP.nnz();
        transposeFineRows = lo_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose fine rows"), localPEntries);
        transposePEntries = size_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose P entries"), localPEntries);
        size_view transposeOffsets(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose offsets"), numCoarseRows);
        if constexpr (executionSpaceIsHost) {
          for (size_t row = 0; row < numCoarseRows; ++row)
            transposeOffsets(row) = transposeRowPtr(row);
          for (size_t fineRow = 0; fineRow < numFineRows; ++fineRow) {
            const auto pRow       = localP.rowConst(static_cast<LO>(fineRow));
            const size_t rowStart = localP.graph.row_map(fineRow);
            for (LO entry = 0; entry < pRow.length; ++entry) {
              const size_t coarseRow = static_cast<size_t>(pRow.colidx(entry));
              const size_t offset    = transposeOffsets(coarseRow)++;
              transposeFineRows(offset) = static_cast<LO>(fineRow);
              transposePEntries(offset) = rowStart + static_cast<size_t>(entry);
            }
          }
        } else {
          Kokkos::parallel_for(
              "StructuredRAP: initialize transpose offsets", range_policy(executionSpace, 0, numCoarseRows),
              KOKKOS_LAMBDA(const size_t row) {
                transposeOffsets(row) = transposeRowPtr(row);
              });
          Kokkos::parallel_for(
              "StructuredRAP: fill transpose adjacency", range_policy(executionSpace, 0, numFineRows),
              KOKKOS_LAMBDA(const size_t fineRow) {
                const auto pRow       = localP.rowConst(static_cast<LO>(fineRow));
                const size_t rowStart = localP.graph.row_map(fineRow);
                for (LO entry = 0; entry < pRow.length; ++entry) {
                  const size_t coarseRow = static_cast<size_t>(pRow.colidx(entry));
                  const size_t offset    = Kokkos::atomic_fetch_add(&transposeOffsets(coarseRow), size_t(1));
                  transposeFineRows(offset) = static_cast<LO>(fineRow);
                  transposePEntries(offset) = rowStart + static_cast<size_t>(entry);
                }
              });
        }
        executionSpace.fence("StructuredRAP: transpose adjacency complete");
      }

      // The imported P and overlapping Ac may use different local column-map
      // orderings.  Translate once outside the product traversal.
      Teuchos::RCP<Teuchos::TimeMonitor> mapTimer = Teuchos::rcp(
          new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "map columns")));
      pToAcColumnMapIsIdentity = ghostedP->getColMap()->isSameAs(*overlapAc->getColMap());
      if (!pToAcColumnMapIsIdentity) {
        const auto ghostedPColMap    = ghostedP->getColMap()->getLocalMap();
        const auto overlapAcColMap   = overlapAc->getColMap()->getLocalMap();
        const size_t numGhostedPCols = ghostedP->getColMap()->getLocalNumElements();
        ghostedPToAcCol = lo_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: P-to-Ac column map"), numGhostedPCols);
        Kokkos::parallel_for(
            "StructuredRAP: map P columns to Ac columns", range_policy(executionSpace, 0, numGhostedPCols),
            KOKKOS_LAMBDA(const size_t column) {
              const GO gid            = ghostedPColMap.getGlobalElement(static_cast<LO>(column));
              ghostedPToAcCol(column) = overlapAcColMap.getLocalElement(gid);
            });
        executionSpace.fence("StructuredRAP: column mapping complete");
      }
      mapTimer = Teuchos::null;

      {
        Teuchos::TimeMonitor timer(
            *Teuchos::TimeMonitor::getNewTimer(timerPrefix + "build geometric interpolation"));

        const size_t localPEntries       = localP.nnz();
        const size_t numLocalPColumns    = P.getColMap()->getLocalNumElements();
        const size_t numGhostedPRows     = localGhostedP.numRows();
        const size_t numGhostedPEntries  = localGhostedP.nnz();
        const size_t numGhostedPColumns  = ghostedP->getColMap()->getLocalNumElements();
        TEUCHOS_TEST_FOR_EXCEPTION(numLocalPColumns % static_cast<size_t>(dofsPerNode) != 0 ||
                                       numGhostedPColumns % static_cast<size_t>(dofsPerNode) != 0,
                                   std::runtime_error,
                                   prefix << "geometric P column maps are not grouped by node and degree of freedom.");

        cache->localGeometricPValues = scalar_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: local geometric P values"),
            localPEntries);
        cache->localGeometricRows = int_view("StructuredRAP: local geometric P rows", numFineRows);

        const size_t numLocalCoarseNodeColumns =
            numLocalPColumns / static_cast<size_t>(dofsPerNode);
        lo_view localCoarseAnchors("StructuredRAP: local coarse anchors",
                                   numLocalCoarseNodeColumns);
        Kokkos::deep_copy(executionSpace, localCoarseAnchors, invalidLO);

        if (interpolationOrder == 1) {
          Kokkos::parallel_for(
              "StructuredRAP: find local coarse anchors",
              range_policy(executionSpace, 0, numFineNodes),
              KOKKOS_LAMBDA(const size_t fineNode) {
                const size_t row = fineNode * static_cast<size_t>(dofsPerNode);
                const auto pRow  = localP.rowConst(static_cast<LO>(row));
                if (pRow.length == 1 && pRow.colidx(0) % dofsPerNode == 0) {
                  const size_t coarseNode =
                      static_cast<size_t>(pRow.colidx(0) / dofsPerNode);
                  localCoarseAnchors(coarseNode) = static_cast<LO>(fineNode);
                }
              });
        }

        const auto localGeometricPValues = cache->localGeometricPValues;
        const auto localGeometricRows    = cache->localGeometricRows;
        Kokkos::parallel_for(
            "StructuredRAP: derive local geometric P",
            range_policy(executionSpace, 0, numFineRows),
            KOKKOS_LAMBDA(const size_t row) {
              const auto pRow       = localP.rowConst(static_cast<LO>(row));
              const size_t rowStart = localP.graph.row_map(row);
              const LO dof          = static_cast<LO>(row % static_cast<size_t>(dofsPerNode));
              const LO fineNode     = static_cast<LO>(row / static_cast<size_t>(dofsPerNode));
              bool geometric        = pRow.length > 0;

              if (interpolationOrder == 0) {
                geometric = pRow.length == 1;
                if (geometric)
                  localGeometricPValues(rowStart) = one;
              } else {
                LO fineCoord[3];
                fineCoord[0] = fineNode % localFineNodes[0];
                fineCoord[1] = (fineNode / localFineNodes[0]) % localFineNodes[1];
                fineCoord[2] = fineNode / (localFineNodes[0] * localFineNodes[1]);
                LO low[3]    = {invalidLO, invalidLO, invalidLO};
                LO high[3]   = {LO(0), LO(0), LO(0)};

                for (LO entry = 0; entry < pRow.length && geometric; ++entry) {
                  const LO column = pRow.colidx(entry);
                  geometric       = column % dofsPerNode == dof;
                  const size_t coarseNode =
                      static_cast<size_t>(column / dofsPerNode);
                  const LO anchor = geometric ? localCoarseAnchors(coarseNode) : invalidLO;
                  geometric       = anchor != invalidLO;
                  if (geometric) {
                    const LO anchorCoord[3] = {
                        anchor % localFineNodes[0],
                        (anchor / localFineNodes[0]) % localFineNodes[1],
                        anchor / (localFineNodes[0] * localFineNodes[1])};
                    for (int dim = 0; dim < numDimensions; ++dim) {
                      if (low[dim] == invalidLO || anchorCoord[dim] < low[dim])
                        low[dim] = anchorCoord[dim];
                      if (anchorCoord[dim] > high[dim])
                        high[dim] = anchorCoord[dim];
                    }
                  }
                }

                for (int dim = 0; dim < numDimensions && geometric; ++dim)
                  geometric = fineCoord[dim] >= low[dim] && fineCoord[dim] <= high[dim];

                for (LO entry = 0; entry < pRow.length && geometric; ++entry) {
                  const LO column = pRow.colidx(entry);
                  const LO anchor =
                      localCoarseAnchors(static_cast<size_t>(column / dofsPerNode));
                  const LO anchorCoord[3] = {
                      anchor % localFineNodes[0],
                      (anchor / localFineNodes[0]) % localFineNodes[1],
                      anchor / (localFineNodes[0] * localFineNodes[1])};
                  SC weight = one;
                  for (int dim = 0; dim < numDimensions; ++dim) {
                    if (high[dim] != low[dim]) {
                      const LO numerator = anchorCoord[dim] == low[dim]
                                               ? high[dim] - fineCoord[dim]
                                               : fineCoord[dim] - low[dim];
                      geometric = geometric &&
                                  (anchorCoord[dim] == low[dim] ||
                                   anchorCoord[dim] == high[dim]);
                      weight *= static_cast<SC>(numerator) /
                                static_cast<SC>(high[dim] - low[dim]);
                    }
                  }
                  localGeometricPValues(rowStart + static_cast<size_t>(entry)) = weight;
                }
              }
              localGeometricRows(row) = geometric ? 1 : 0;
            });

        // The row structure is already immutable and remains alive through
        // importedP. Alias it rather than allocating and copying it.
        cache->ghostedGeometricRowPtr = localGhostedP.graph.row_map;
        lo_view mappedGhostedGeometricColumns;
        if (pToAcColumnMapIsIdentity) {
          cache->ghostedGeometricColumns = localGhostedP.graph.entries;
        } else {
          mappedGhostedGeometricColumns = lo_view(
              Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: mapped geometric columns"),
              numGhostedPEntries);
          cache->ghostedGeometricColumns = mappedGhostedGeometricColumns;
        }

        const bool reuseLocalGeometricData =
            ghostedP.getRawPtr() == &P &&
            numGhostedPRows == numFineRows &&
            numGhostedPEntries == localPEntries;
        if (reuseLocalGeometricData) {
          // With no imported P rows, the right-hand P is exactly the local P
          // already processed above. Share its weights and row classifications
          // instead of repeating anchor discovery and interpolation.
          cache->ghostedGeometricPValues = cache->localGeometricPValues;
          cache->ghostedGeometricRows    = cache->localGeometricRows;
          if (!pToAcColumnMapIsIdentity) {
            Kokkos::parallel_for(
                "StructuredRAP: map local geometric columns",
                range_policy(executionSpace, 0, numGhostedPEntries),
                KOKKOS_LAMBDA(const size_t entry) {
                  const LO pColumn = localP.graph.entries(entry);
                  mappedGhostedGeometricColumns(entry) =
                      ghostedPToAcCol(static_cast<size_t>(pColumn));
                });
          }
        } else {
          cache->ghostedGeometricPValues = scalar_view(
              Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: ghosted geometric P values"),
              numGhostedPEntries);
          cache->ghostedGeometricRows =
              int_view("StructuredRAP: ghosted geometric P rows", numGhostedPRows);

          const size_t numGhostedCoarseNodeColumns =
              numGhostedPColumns / static_cast<size_t>(dofsPerNode);
          lo_view ghostedCoarseAnchors("StructuredRAP: ghosted coarse anchors",
                                       numGhostedCoarseNodeColumns);
          Kokkos::deep_copy(executionSpace, ghostedCoarseAnchors, invalidLO);

          const auto ghostedPRowMap = ghostedP->getRowMap()->getLocalMap();
          const auto localPRowMap   = P.getRowMap()->getLocalMap();
          if (interpolationOrder == 1) {
            Kokkos::parallel_for(
                "StructuredRAP: find ghosted coarse anchors",
                range_policy(executionSpace, 0, numGhostedPRows),
                KOKKOS_LAMBDA(const size_t ghostedRow) {
                  const GO gid       = ghostedPRowMap.getGlobalElement(static_cast<LO>(ghostedRow));
                  const LO localRow  = localPRowMap.getLocalElement(gid);
                  const auto pRow    = localGhostedP.rowConst(static_cast<LO>(ghostedRow));
                  if (localRow != invalidLO && localRow % dofsPerNode == 0 &&
                      pRow.length == 1 && pRow.colidx(0) % dofsPerNode == 0) {
                    const size_t coarseNode =
                        static_cast<size_t>(pRow.colidx(0) / dofsPerNode);
                    ghostedCoarseAnchors(coarseNode) = localRow / dofsPerNode;
                  }
                });
          }

          const auto ghostedGeometricPValues = cache->ghostedGeometricPValues;
          const auto ghostedGeometricRows    = cache->ghostedGeometricRows;
          Kokkos::parallel_for(
              "StructuredRAP: derive ghosted geometric P",
              range_policy(executionSpace, 0, numGhostedPRows),
              KOKKOS_LAMBDA(const size_t ghostedRow) {
                const auto pRow       = localGhostedP.rowConst(static_cast<LO>(ghostedRow));
                const size_t rowStart = localGhostedP.graph.row_map(ghostedRow);
                const GO gid          = ghostedPRowMap.getGlobalElement(static_cast<LO>(ghostedRow));
                const LO localRow     = localPRowMap.getLocalElement(gid);
                bool geometric        = interpolationOrder == 0
                                            ? pRow.length == 1
                                            : localRow != invalidLO && pRow.length > 0;
                LO fineCoord[3]       = {LO(0), LO(0), LO(0)};
                LO low[3]             = {invalidLO, invalidLO, invalidLO};
                LO high[3]            = {LO(0), LO(0), LO(0)};

              if (interpolationOrder == 1 && geometric) {
                const LO fineNode = localRow / dofsPerNode;
                fineCoord[0]      = fineNode % localFineNodes[0];
                fineCoord[1]      = (fineNode / localFineNodes[0]) % localFineNodes[1];
                fineCoord[2]      = fineNode / (localFineNodes[0] * localFineNodes[1]);
                const LO dof      = localRow % dofsPerNode;
                for (LO entry = 0; entry < pRow.length && geometric; ++entry) {
                  const LO column = pRow.colidx(entry);
                  geometric       = column % dofsPerNode == dof;
                  const LO anchor = geometric
                                        ? ghostedCoarseAnchors(
                                              static_cast<size_t>(column / dofsPerNode))
                                        : invalidLO;
                  geometric = anchor != invalidLO;
                  if (geometric) {
                    const LO anchorCoord[3] = {
                        anchor % localFineNodes[0],
                        (anchor / localFineNodes[0]) % localFineNodes[1],
                        anchor / (localFineNodes[0] * localFineNodes[1])};
                    for (int dim = 0; dim < numDimensions; ++dim) {
                      if (low[dim] == invalidLO || anchorCoord[dim] < low[dim])
                        low[dim] = anchorCoord[dim];
                      if (anchorCoord[dim] > high[dim])
                        high[dim] = anchorCoord[dim];
                    }
                  }
                }
                for (int dim = 0; dim < numDimensions && geometric; ++dim)
                  geometric = fineCoord[dim] >= low[dim] && fineCoord[dim] <= high[dim];
              }

              for (LO entry = 0; entry < pRow.length; ++entry) {
                const LO pColumn = pRow.colidx(entry);
                const LO acColumn = pToAcColumnMapIsIdentity
                                        ? pColumn
                                        : ghostedPToAcCol(static_cast<size_t>(pColumn));
                if (!pToAcColumnMapIsIdentity)
                  mappedGhostedGeometricColumns(rowStart + static_cast<size_t>(entry)) = acColumn;
                SC weight = one;
                if (interpolationOrder == 1 && geometric) {
                  const LO anchor = ghostedCoarseAnchors(
                      static_cast<size_t>(pColumn / dofsPerNode));
                  const LO anchorCoord[3] = {
                      anchor % localFineNodes[0],
                      (anchor / localFineNodes[0]) % localFineNodes[1],
                      anchor / (localFineNodes[0] * localFineNodes[1])};
                  for (int dim = 0; dim < numDimensions; ++dim) {
                    if (high[dim] != low[dim]) {
                      const LO numerator = anchorCoord[dim] == low[dim]
                                               ? high[dim] - fineCoord[dim]
                                               : fineCoord[dim] - low[dim];
                      geometric = geometric &&
                                  (anchorCoord[dim] == low[dim] ||
                                   anchorCoord[dim] == high[dim]);
                      weight *= static_cast<SC>(numerator) /
                                static_cast<SC>(high[dim] - low[dim]);
                    }
                  }
                }
                ghostedGeometricPValues(rowStart + static_cast<size_t>(entry)) = weight;
              }
              ghostedGeometricRows(ghostedRow) = geometric ? 1 : 0;
              });
        }
        executionSpace.fence("StructuredRAP: geometric interpolation cache complete");
      }

      cache->transposeRowPtr          = transposeRowPtr;
      cache->transposeFineRows        = transposeFineRows;
      cache->transposePEntries        = transposePEntries;
      cache->ghostedPToAcCol          = ghostedPToAcCol;
      cache->pToAcColumnMapIsIdentity = pToAcColumnMapIsIdentity;
      cache->maxAcRowEntries          = overlapAc->getLocalMaxNumRowEntries();
    } else {
      transposeRowPtr          = cache->transposeRowPtr;
      transposeFineRows        = cache->transposeFineRows;
      transposePEntries        = cache->transposePEntries;
      ghostedPToAcCol          = cache->ghostedPToAcCol;
      pToAcColumnMapIsIdentity = cache->pToAcColumnMapIsIdentity;
    }

    const auto localGeometricPValues   = cache->localGeometricPValues;
    const auto localGeometricRows      = cache->localGeometricRows;
    const auto ghostedGeometricRowPtr  = cache->ghostedGeometricRowPtr;
    const auto ghostedGeometricColumns = cache->ghostedGeometricColumns;
    const auto ghostedGeometricPValues = cache->ghostedGeometricPValues;
    const auto ghostedGeometricRows    = cache->ghostedGeometricRows;

    Teuchos::RCP<Teuchos::TimeMonitor> numericTimer = Teuchos::rcp(
        new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "numeric")));
    if constexpr (executionSpaceIsHost) {
        // Match Tpetra's CPU RAP strategy: each worker owns a contiguous set
        // of coarse rows and uses a dense column-to-entry table for O(1)
        // insertion into the supplied Ac graph.
        const size_t numAcColumns = overlapAc->getColMap()->getLocalNumElements();
        const size_t numWorkers   = static_cast<size_t>(executionSpace.concurrency());
        const size_t invalidEntry = Teuchos::OrdinalTraits<size_t>::invalid();
        size_view acEntryLookup(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: CPU Ac entry lookup"),
            numWorkers * numAcColumns);
        Kokkos::deep_copy(executionSpace, acEntryLookup, invalidEntry);

        Kokkos::parallel_for(
            "StructuredRAP: CPU row P^T A P", range_policy(executionSpace, 0, numWorkers),
            KOKKOS_LAMBDA(const size_t worker) {
              const size_t firstCoarseRow = worker * numCoarseRows / numWorkers;
              const size_t lastCoarseRow  = (worker + 1) * numCoarseRows / numWorkers;
              const size_t lookupOffset   = worker * numAcColumns;

              for (size_t coarseRow = firstCoarseRow; coarseRow < lastCoarseRow; ++coarseRow) {
                const size_t acBegin = localOverlapAc.graph.row_map(coarseRow);
                const size_t acEnd   = localOverlapAc.graph.row_map(coarseRow + 1);
                for (size_t acEntry = acBegin; acEntry < acEnd; ++acEntry) {
                  const LO acColumn = localOverlapAc.graph.entries(acEntry);
                  localOverlapAc.values(acEntry) = zero;
                  acEntryLookup(lookupOffset + static_cast<size_t>(acColumn)) = acEntry;
                }

                const size_t begin = transposeRowPtr(coarseRow);
                const size_t end   = transposeRowPtr(coarseRow + 1);
                for (size_t transposeEntry = begin; transposeEntry < end; ++transposeEntry) {
                  const LO fineRow        = transposeFineRows(transposeEntry);
                  const size_t localPEntry = transposePEntries(transposeEntry);
                  const SC restriction    = localGeometricRows(static_cast<size_t>(fineRow))
                                                ? localGeometricPValues(localPEntry)
                                                : localP.values(localPEntry);
                  const auto aRow = localA.rowConst(fineRow);

                  for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                    const SC raValue       = restriction * aRow.value(aEntry);
                    const LO ghostedPRow   = aRow.colidx(aEntry);
                    const size_t pBegin    = ghostedGeometricRowPtr(static_cast<size_t>(ghostedPRow));
                    const size_t pEnd      = ghostedGeometricRowPtr(static_cast<size_t>(ghostedPRow) + 1);
                    const bool geometricP = ghostedGeometricRows(static_cast<size_t>(ghostedPRow)) != 0;
                    for (size_t pEntry = pBegin; pEntry < pEnd; ++pEntry) {
                      const LO acColumn = ghostedGeometricColumns(pEntry);
                      if (acColumn == invalidLO) {
                        Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                        continue;
                      }

                      const size_t acEntry =
                          acEntryLookup(lookupOffset + static_cast<size_t>(acColumn));
                      if (acEntry < acBegin || acEntry >= acEnd) {
                        Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                        continue;
                      }
                      const SC pValue = geometricP
                                            ? ghostedGeometricPValues(pEntry)
                                            : localGhostedP.values(pEntry);
                      localOverlapAc.values(acEntry) += raValue * pValue;
                    }
                  }
                }
              }
            });
    } else {
      // Short transpose rows are cheapest with one thread per coarse row.  The
      // vector lanes cooperate on the innermost P row, which is the contiguous
      // traversal in this product.
      constexpr size_t teamRowThreshold = 8;
      constexpr int vectorLength        = 8;
      team_policy policy(executionSpace, numCoarseRows, Kokkos::AUTO(), vectorLength);
      const size_t scratchBytes = scratch_view::shmem_size(cache->maxAcRowEntries);
      int scratchLevel          = -1;
      if (scratchBytes <= static_cast<size_t>(policy.scratch_size_max(0))) {
        scratchLevel = 0;
        policy.set_scratch_size(scratchLevel, Kokkos::PerTeam(scratchBytes));
      } else if (scratchBytes <= static_cast<size_t>(policy.scratch_size_max(1))) {
        scratchLevel = 1;
        policy.set_scratch_size(scratchLevel, Kokkos::PerTeam(scratchBytes));
      }
      const bool teamScratchEnabled = scratchLevel >= 0;

      const team_policy rowPolicy(executionSpace, numCoarseRows, 1, vectorLength);
      Kokkos::parallel_for(
          "StructuredRAP: row-thread P^T A P", rowPolicy,
          KOKKOS_LAMBDA(const member_type& team) {
            const size_t coarseRow = static_cast<size_t>(team.league_rank());
            const size_t begin     = transposeRowPtr(coarseRow);
            const size_t end   = transposeRowPtr(coarseRow + 1);
            if (teamScratchEnabled && end - begin >= teamRowThreshold)
              return;

            const size_t acBegin = localOverlapAc.graph.row_map(coarseRow);
            const size_t acEnd   = localOverlapAc.graph.row_map(coarseRow + 1);
            for (size_t transposeEntry = begin; transposeEntry < end; ++transposeEntry) {
              const LO fineRow        = transposeFineRows(transposeEntry);
              const size_t localPEntry = transposePEntries(transposeEntry);
              const SC restriction    = localGeometricRows(static_cast<size_t>(fineRow))
                                            ? localGeometricPValues(localPEntry)
                                            : localP.values(localPEntry);
              const auto aRow = localA.rowConst(fineRow);

              for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                const LO ghostedPRow   = aRow.colidx(aEntry);
                const SC raValue       = restriction * aRow.value(aEntry);
                const size_t pBegin    = ghostedGeometricRowPtr(static_cast<size_t>(ghostedPRow));
                const size_t pEnd      = ghostedGeometricRowPtr(static_cast<size_t>(ghostedPRow) + 1);
                const bool geometricP = ghostedGeometricRows(static_cast<size_t>(ghostedPRow)) != 0;

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, pEnd - pBegin),
                    [&](const size_t pOffset) {
                      const size_t pEntry = pBegin + pOffset;
                      const LO acColumn   = ghostedGeometricColumns(pEntry);
                      if (acColumn == invalidLO) {
                        Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                        return;
                      }

                      size_t lower = acBegin;
                      size_t upper = acEnd;
                      while (lower < upper) {
                        const size_t middle = lower + (upper - lower) / 2;
                        if (localOverlapAc.graph.entries(middle) < acColumn)
                          lower = middle + 1;
                        else
                          upper = middle;
                      }

                      if (lower == acEnd || localOverlapAc.graph.entries(lower) != acColumn) {
                        Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                        return;
                      }
                      const SC pValue = geometricP
                                            ? ghostedGeometricPValues(pEntry)
                                            : localGhostedP.values(pEntry);
                      localOverlapAc.values(lower) += raValue * pValue;
                    });
              }
            }
          });

      // Heavy rows use a full team and accumulate into row-sized team scratch,
      // avoiding global atomics while retaining parallelism across P^T entries.
      if (teamScratchEnabled) {
        Kokkos::parallel_for(
            "StructuredRAP: fused P^T A P", policy,
            KOKKOS_LAMBDA(const member_type& team) {
              const size_t coarseRow = static_cast<size_t>(team.league_rank());
              const size_t begin     = transposeRowPtr(coarseRow);
              const size_t end   = transposeRowPtr(coarseRow + 1);
              if (end - begin < teamRowThreshold)
                return;

              const size_t acBegin   = localOverlapAc.graph.row_map(coarseRow);
              const size_t acEnd     = localOverlapAc.graph.row_map(coarseRow + 1);
              const size_t acEntries = acEnd - acBegin;
              scratch_view rowValues(team.team_scratch(scratchLevel), acEntries);

              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, acEntries),
                                   [&](const size_t entry) { rowValues(entry) = zero; });
              team.team_barrier();

              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team, begin, end),
                  [&](const size_t transposeEntry) {
                    const LO fineRow        = transposeFineRows(transposeEntry);
                    const size_t localPEntry = transposePEntries(transposeEntry);
                    const SC restriction    = localGeometricRows(static_cast<size_t>(fineRow))
                                                  ? localGeometricPValues(localPEntry)
                                                  : localP.values(localPEntry);
                    const auto aRow = localA.rowConst(fineRow);

                    for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                      const LO ghostedPRow   = aRow.colidx(aEntry);
                      const SC raValue       = restriction * aRow.value(aEntry);
                      const size_t pBegin    = ghostedGeometricRowPtr(static_cast<size_t>(ghostedPRow));
                      const size_t pEnd      = ghostedGeometricRowPtr(static_cast<size_t>(ghostedPRow) + 1);
                      const bool geometricP = ghostedGeometricRows(static_cast<size_t>(ghostedPRow)) != 0;

                      Kokkos::parallel_for(
                          Kokkos::ThreadVectorRange(team, pEnd - pBegin),
                          [&](const size_t pOffset) {
                            const size_t pEntry = pBegin + pOffset;
                            const LO acColumn   = ghostedGeometricColumns(pEntry);
                            if (acColumn == invalidLO) {
                              Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                              return;
                            }

                            size_t lower = acBegin;
                            size_t upper = acEnd;
                            while (lower < upper) {
                              const size_t middle = lower + (upper - lower) / 2;
                              if (localOverlapAc.graph.entries(middle) < acColumn)
                                lower = middle + 1;
                              else
                                upper = middle;
                            }

                            if (lower == acEnd || localOverlapAc.graph.entries(lower) != acColumn) {
                              Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                              return;
                            }

                            const SC pValue = geometricP
                                                  ? ghostedGeometricPValues(pEntry)
                                                  : localGhostedP.values(pEntry);
                            Kokkos::atomic_add(&rowValues(lower - acBegin), raValue * pValue);
                          });
                    }
                  });
              team.team_barrier();
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team, acEntries),
                  [&](const size_t entry) { localOverlapAc.values(acBegin + entry) = rowValues(entry); });
            });
      }
    }
    executionSpace.fence("StructuredRAP: fused P^T A P complete");
    numericTimer = Teuchos::null;

    int missingGraphEntryHost = 0;
    Kokkos::deep_copy(executionSpace, missingGraphEntryHost, missingGraphEntry);
    TEUCHOS_TEST_FOR_EXCEPTION(missingGraphEntryHost != 0,
                               std::runtime_error,
                               prefix << "the supplied Ac graph does not contain every entry generated by P^T A P.");

    if (!coarseRowImporter.is_null()) {
      Teuchos::TimeMonitor timer(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "export Ac"));
      Ac.doExport(*overlapAc, *coarseRowImporter, Xpetra::ADD);
    }

    // MatrixFactory::Build(graph) creates a fill-active matrix even though its
    // static graph is already fill complete.  Match the state produced by the
    // Xpetra triple-product path before returning it to StructuredRAPFactory.
    if (!Ac.isFillComplete()) {
      Teuchos::TimeMonitor timer(*Teuchos::TimeMonitor::getNewTimer(timerPrefix + "fillComplete"));
      Ac.fillComplete(P.getDomainMap(), P.getDomainMap(), params);
    }

    if (!params.is_null())
      params->set(cacheKey(), cache);
  }
};

}  // namespace Details
}  // namespace MueLu

#endif  // MUELU_STRUCTUREDRAPKERNEL_HPP
