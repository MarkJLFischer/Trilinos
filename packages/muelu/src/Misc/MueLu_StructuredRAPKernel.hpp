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

#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Xpetra_Matrix.hpp>

namespace MueLu {
namespace Details {

/**
 * Specialized numeric kernel for the serial host Elasticity2D stencil.
 *
 * The implemented path requires two dimensions, two degrees of freedom per
 * node, order-zero interpolation, and the full radius-one coarse stencil.
 * Unsupported configurations are rejected; no generic sparse RAP fallback is
 * provided.
 */
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class StructuredRAPKernel {
 public:
  using Matrix = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

 private:
  using device_type = typename Node::device_type;
  using lo_view     = Kokkos::View<LocalOrdinal*, device_type>;

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
    using range_policy    = Kokkos::RangePolicy<execution_space, Kokkos::IndexType<size_t>>;
    static constexpr bool executionSpaceIsHost =
        std::is_same<typename device_type::memory_space, Kokkos::HostSpace>::value;

    const std::string prefix      = "MueLu::Details::StructuredRAPKernel::Compute: ";
    const std::string timerPrefix = "MueLu: StructuredRAPKernel: ";

    const int numDimensions = fineStencil.numDimensions;
    const LO dofsPerNode     = fineStencil.dofsPerNode;

    TEUCHOS_TEST_FOR_EXCEPTION(!executionSpaceIsHost,
                               std::runtime_error,
                               prefix << "the current structured specialization requires host memory.");
    TEUCHOS_TEST_FOR_EXCEPTION(interpolationOrder != 0,
                               std::runtime_error,
                               prefix << "the current structured specialization requires order-zero interpolation.");
    TEUCHOS_TEST_FOR_EXCEPTION(numDimensions != 2,
                               std::runtime_error,
                               prefix << "the current structured specialization requires two dimensions.");
    TEUCHOS_TEST_FOR_EXCEPTION(fineStencil.dofsPerNode != LocalOrdinal(2),
                               std::runtime_error,
                               prefix << "the current structured specialization requires two DOFs per node.");
    TEUCHOS_TEST_FOR_EXCEPTION(fineStencil.stencilOffsets.size() != size_t(9),
                               std::runtime_error,
                               prefix << "the current structured specialization requires a full nine-point stencil.");
    TEUCHOS_TEST_FOR_EXCEPTION(fineStencil.entries.size() != size_t(36),
                               std::runtime_error,
                               prefix << "the current structured specialization requires 36 scalar stencil entries.");
    TEUCHOS_TEST_FOR_EXCEPTION(lFineNodesPerDim.size() < 2 || lCoarseNodesPerDim.size() < 2,
                               std::runtime_error,
                               prefix << "two fine- and coarse-grid dimensions are required.");
    TEUCHOS_TEST_FOR_EXCEPTION(structuredCoarseningRate.empty(),
                               std::runtime_error,
                               prefix << "the Elasticity2D specialization requires a coarsening rate.");
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
      const bool useDetectedStencil =
          interpolationOrder == 0 &&
          numDimensions == 2 &&
          fineStencil.dofsPerNode == LO(2) &&
          fineStencil.stencilOffsets.size() == size_t(9) &&
          fineStencil.entries.size() == size_t(36) &&
          structuredDimensionsMatch &&
          A.GetFixedBlockSize() == 2 &&
          localCoarseNodes[0] >= LO(3) &&
          A.getRowMap()->getComm()->getSize() == 1 &&
          A.getColMap()->isSameAs(*A.getRowMap()) &&
          Ac.getColMap()->isSameAs(*Ac.getRowMap()) &&
          A.getRowMap()->getLocalNumElements() == expectedFineRows &&
          Ac.getRowMap()->getLocalNumElements() == expectedCoarseRows;

      if (useDetectedStencil) {
        const SC zero  = Teuchos::ScalarTraits<SC>::zero();
        const auto localA = A.getLocalMatrixDevice();
        auto localAc      = Ac.getLocalMatrixDevice();

        const size_t numFineStencilEntries = fineStencil.entries.size();
        Kokkos::View<int*, device_type> stencilOffsetX(
            "StructuredRAP: fine stencil X offsets", numFineStencilEntries);
        Kokkos::View<int*, device_type> stencilOffsetY(
            "StructuredRAP: fine stencil Y offsets", numFineStencilEntries);
        lo_view stencilRowDof(
            "StructuredRAP: fine stencil row DOFs", numFineStencilEntries);
        lo_view stencilColumnDof(
            "StructuredRAP: fine stencil column DOFs", numFineStencilEntries);
        lo_view stencilEntryOrdinal(
            "StructuredRAP: fine stencil entry ordinals", numFineStencilEntries);

        auto stencilOffsetXHost = Kokkos::create_mirror_view(stencilOffsetX);
        auto stencilOffsetYHost = Kokkos::create_mirror_view(stencilOffsetY);
        auto stencilRowDofHost = Kokkos::create_mirror_view(stencilRowDof);
        auto stencilColumnDofHost = Kokkos::create_mirror_view(stencilColumnDof);
        auto stencilEntryOrdinalHost = Kokkos::create_mirror_view(stencilEntryOrdinal);
        for (size_t entry = 0; entry < numFineStencilEntries; ++entry) {
          stencilOffsetXHost(entry) = fineStencil.entries[entry].offset.x;
          stencilOffsetYHost(entry) = fineStencil.entries[entry].offset.y;
          stencilRowDofHost(entry) = fineStencil.entries[entry].rowDof;
          stencilColumnDofHost(entry) = fineStencil.entries[entry].columnDof;
          stencilEntryOrdinalHost(entry) = fineStencil.entries[entry].entryOrdinal;
        }
        Kokkos::deep_copy(executionSpace, stencilOffsetX, stencilOffsetXHost);
        Kokkos::deep_copy(executionSpace, stencilOffsetY, stencilOffsetYHost);
        Kokkos::deep_copy(executionSpace, stencilRowDof, stencilRowDofHost);
        Kokkos::deep_copy(executionSpace, stencilColumnDof, stencilColumnDofHost);
        Kokkos::deep_copy(executionSpace, stencilEntryOrdinal, stencilEntryOrdinalHost);
        const size_t numCoarseNodes =
            static_cast<size_t>(localCoarseNodes[0]) *
            static_cast<size_t>(localCoarseNodes[1]);
        const size_t numWorkers =
            static_cast<size_t>(executionSpace.concurrency());

        // Build cache-resident one-dimensional ownership maps once.
        lo_view fineXToCoarseX(
            Kokkos::ViewAllocateWithoutInitializing(
                "StructuredRAP: fine-X to coarse-X map"),
            static_cast<size_t>(localFineNodes[0]));
        lo_view fineYToCoarseY(
            Kokkos::ViewAllocateWithoutInitializing(
                "StructuredRAP: fine-Y to coarse-Y map"),
            static_cast<size_t>(localFineNodes[1]));
        {
          Teuchos::TimeMonitor timer(
              *Teuchos::TimeMonitor::getNewTimer(
                  timerPrefix + "build constant interpolation maps"));
          Kokkos::parallel_for(
              "StructuredRAP: build fine-X to coarse-X map",
              range_policy(executionSpace, 0,
                           static_cast<size_t>(localFineNodes[0])),
              KOKKOS_LAMBDA(const size_t fineXIndex) {
                const LO fineX = static_cast<LO>(fineXIndex);
                fineXToCoarseX(fineXIndex) = constantInterpolationCoarse(
                    fineX, localFineNodes[0], coarseningRate[0]);
              });
          Kokkos::parallel_for(
              "StructuredRAP: build fine-Y to coarse-Y map",
              range_policy(executionSpace, 0,
                           static_cast<size_t>(localFineNodes[1])),
              KOKKOS_LAMBDA(const size_t fineYIndex) {
                const LO fineY = static_cast<LO>(fineYIndex);
                fineYToCoarseY(fineYIndex) = constantInterpolationCoarse(
                    fineY, localFineNodes[1], coarseningRate[1]);
              });
          executionSpace.fence(
              "StructuredRAP: constant interpolation maps complete");
        }

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
                  const LO stencilWidth = lastStencilX - firstStencilX + 1;

                  for (LO fineY = fineYBegin; fineY <= fineYEnd;
                       ++fineY) {
                    for (LO fineX = fineXBegin; fineX <= fineXEnd;
                         ++fineX) {
                      const LO fineNode =
                          fineY * localFineNodes[0] + fineX;
                      const LO firstFineRow = fineNode * dofsPerNode;
                      const bool interior =
                          fineX > 0 && fineX + 1 < localFineNodes[0] &&
                          fineY > 0 && fineY + 1 < localFineNodes[1];

                      for (size_t stencilEntry = 0;
                           stencilEntry < numFineStencilEntries;
                           ++stencilEntry) {
                        const LO neighborX =
                            fineX + static_cast<LO>(stencilOffsetX(stencilEntry));
                        const LO neighborY =
                            fineY + static_cast<LO>(stencilOffsetY(stencilEntry));
                        if (neighborX < LO(0) || neighborX >= localFineNodes[0] ||
                            neighborY < LO(0) || neighborY >= localFineNodes[1])
                          continue;

                        const LO rowDof = stencilRowDof(stencilEntry);
                        const LO columnDof = stencilColumnDof(stencilEntry);
                        const LO targetX = fineXToCoarseX(
                            static_cast<size_t>(neighborX));
                        const LO targetY = fineYToCoarseY(
                            static_cast<size_t>(neighborY));
                        const size_t coarseNodeOffset = static_cast<size_t>(
                            (targetY - firstStencilY) * stencilWidth +
                            (targetX - firstStencilX)) * size_t(2);
                        const size_t acRowBegin =
                            localAc.graph.row_map(firstCoarseRow +
                                                  static_cast<size_t>(rowDof));
                        const size_t acEntry =
                            acRowBegin + coarseNodeOffset +
                            static_cast<size_t>(columnDof);

                        const size_t aRow =
                            static_cast<size_t>(firstFineRow + rowDof);
                        const size_t aRowBegin =
                            localA.graph.row_map(aRow);
                        size_t aEntry =
                            aRowBegin +
                            static_cast<size_t>(stencilEntryOrdinal(stencilEntry));

                        if (!interior) {
                          const LO neighborNode =
                              neighborY * localFineNodes[0] + neighborX;
                          const LO expectedColumn =
                              neighborNode * dofsPerNode + columnDof;
                          const size_t aRowEnd =
                              localA.graph.row_map(aRow + size_t(1));
                          aEntry = aRowEnd;
                          for (size_t candidate = aRowBegin;
                               candidate < aRowEnd; ++candidate) {
                            if (localA.graph.entries(candidate) == expectedColumn) {
                              aEntry = candidate;
                              break;
                            }
                          }
                          if (aEntry == aRowEnd)
                            continue;
                        }

                        localAc.values(acEntry) += localA.values(aEntry);
                      }
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
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::runtime_error,
        prefix << "only the serial host two-dimensional, two-DOF, order-zero "
                  "nine-point specialization is currently implemented; no generic "
                  "triple-matrix-product fallback is available.");
  }
};

}  // namespace Details
}  // namespace MueLu

#endif  // MUELU_STRUCTUREDRAPKERNEL_HPP
