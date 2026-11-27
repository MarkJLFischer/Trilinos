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
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosKernels_ArithTraits.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

namespace MueLu {
namespace Details {

/**
 * Specialized numeric kernel for one-, two-, and three-dimensional structured RAP.
 *
 * The implemented path supports order-zero and order-one interpolation.
 * Its numeric plan is generated from the detected fine stencil and sparse graphs.
 */
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class StructuredRAPKernel {
 public:
  using Matrix = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

 private:
  using LO          = LocalOrdinal;
  using device_type = typename Node::device_type;
  using lo_view     = Kokkos::View<LocalOrdinal*, device_type>;
  using scalar_view = Kokkos::View<Scalar*, device_type>;
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
    const LO anchor   = coarseAnchor(coarse, numFine, numCoarse, rate);
    const LO previous = coarse == 0 ? LO(0) : coarseAnchor(coarse - 1, numFine, numCoarse, rate);
    const LO next     = coarse + 1 == numCoarse ? numFine - 1
                                                : coarseAnchor(coarse + 1, numFine, numCoarse, rate);
    return {{coarse == 0 ? LO(0) : previous + (anchor - previous) / 2 + 1,
             coarse + 1 == numCoarse ? numFine - 1 : anchor + (next - anchor) / 2}};
  }

  // Inclusive support of a piecewise-linear coarse basis function.
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<LO, 2> linearSupportBounds(LO coarse, LO numFine,
                                                  LO numCoarse, LO rate) {
    return {{coarse == 0 ? LO(0) : coarseAnchor(coarse - 1, numFine, numCoarse, rate),
             coarse + 1 == numCoarse ? numFine - 1
                                     : coarseAnchor(coarse + 1, numFine, numCoarse, rate)}};
  }

  template <class Weight>
  KOKKOS_INLINE_FUNCTION static Weight linearBasisWeight(
      const Kokkos::Array<LO, 3>& fine,
      const Kokkos::Array<LO, 3>& coarse,
      const Kokkos::Array<LO, 3>& fineNodes,
      const Kokkos::Array<LO, 3>& coarseNodes,
      const Kokkos::Array<LO, 3>& rate, const int numDimensions) {
    Weight weight = Weight(1);
    for (int dim = 0; dim < numDimensions; ++dim) {
      const LO anchor = coarseAnchor(coarse[dim], fineNodes[dim], coarseNodes[dim], rate[dim]);
      if (fine[dim] < anchor) {
        if (coarse[dim] == 0)
          return Weight(0);
        const LO previous = coarseAnchor(coarse[dim] - 1, fineNodes[dim], coarseNodes[dim], rate[dim]);
        weight *= Weight(fine[dim] - previous) / Weight(anchor - previous);
      } else if (fine[dim] > anchor) {
        if (coarse[dim] + 1 == coarseNodes[dim])
          return Weight(0);
        const LO next = coarseAnchor(coarse[dim] + 1, fineNodes[dim], coarseNodes[dim], rate[dim]);
        weight *= Weight(next - fine[dim]) / Weight(next - anchor);
      }
    }
    return weight;
  }

  static size_t boundaryClass(LO coarse, LO numCoarse,
                              size_t leftDepth, size_t rightDepth) {
    const size_t left  = static_cast<size_t>(coarse);
    const size_t right = static_cast<size_t>(numCoarse - coarse - 1);
    return left < leftDepth ? left
                            : (right < rightDepth ? leftDepth + rightDepth - right : leftDepth);
  }

  // Keep host mirrors alive until the preparation fence, allowing asynchronous copies.
  template <class View>
  struct PlanArray {
    View device;
    typename View::host_mirror_type host;

    PlanArray(const char* label, size_t size)
      : device(Kokkos::ViewAllocateWithoutInitializing(label), size)
      , host(Kokkos::create_mirror_view(device)) {}

    View copyToDevice(const typename device_type::execution_space& executionSpace) const {
      Kokkos::deep_copy(executionSpace, device, host);
      return device;
    }
  };

 public:
  template <class View>
  static auto gatherToHost(
      const View& source, const std::vector<size_t>& sourceIndices,
      const std::string& label,
      const typename device_type::execution_space& executionSpace)
      -> typename Kokkos::View<
          typename View::non_const_value_type*, device_type>::host_mirror_type {
    using value_type   = typename View::non_const_value_type;
    using range_policy = Kokkos::RangePolicy<
        typename device_type::execution_space, Kokkos::IndexType<size_t>>;
    PlanArray<size_view> indices((label + " indices").c_str(),
                                 sourceIndices.size());
    for (size_t i = 0; i < sourceIndices.size(); ++i)
      indices.host(i) = sourceIndices[i];
    const auto indicesDevice = indices.copyToDevice(executionSpace);
    Kokkos::View<value_type*, device_type> gathered(
        Kokkos::ViewAllocateWithoutInitializing(label), sourceIndices.size());
    Kokkos::parallel_for(
        label, range_policy(executionSpace, 0, sourceIndices.size()),
        KOKKOS_LAMBDA(const size_t i) {
          gathered(i) = source(indicesDevice(i));
        });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gathered);
  }

 private:
  static void sortAndUnique(std::vector<size_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
  }

  template <int InterpolationOrder, class LocalMatrix, class CoefficientView>
  KOKKOS_INLINE_FUNCTION static auto weightedAValue(
      const LocalMatrix& localA, const size_t address,
      const CoefficientView& coefficients, const size_t contribution) {
    auto value = localA.values(address);
    if (InterpolationOrder == 1)
      value *= coefficients(contribution);
    return value;
  }

 public:
  struct AxisClassMap {
    lo_view boundaryClasses;
    LO numCoordinates = 0;
    LO leftCount      = 0;
    LO rightCount     = 0;
    LO interiorClass  = 0;

    KOKKOS_INLINE_FUNCTION
    LO operator()(const LO coordinate) const {
      if (coordinate < leftCount)
        return boundaryClasses(coordinate);
      const LO rightBegin = numCoordinates - rightCount;
      if (coordinate >= rightBegin)
        return boundaryClasses(leftCount + coordinate - rightBegin);
      return interiorClass;
    }

    KOKKOS_INLINE_FUNCTION
    bool isInterior(const LO coordinate) const {
      return coordinate >= leftCount &&
             coordinate < numCoordinates - rightCount;
    }
  };

 private:
  KOKKOS_INLINE_FUNCTION
  static LocalOrdinal constantInterpolationCoarse(
      const LocalOrdinal fine,
      const LocalOrdinal numFine,
      const LocalOrdinal rate) {
    LocalOrdinal endRate = (numFine - 1) % rate;
    if (endRate == 0)
      endRate = rate;

    LocalOrdinal coarse          = fine / rate;
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
    using execution_space         = typename device_type::execution_space;
    using range_policy            = Kokkos::RangePolicy<execution_space,
                                             Kokkos::IndexType<size_t>>;
    const std::string timerPrefix = "MueLu: StructuredRAPKernel: ";

    const int numDimensions = fineStencil.numDimensions;
    const LO dofsPerNode    = fineStencil.dofsPerNode;

    Kokkos::Array<LO, 3> localFineNodes{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    Kokkos::Array<LO, 3> localCoarseNodes{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    Kokkos::Array<LO, 3> coarseningRate{{Teuchos::as<LO>(1), Teuchos::as<LO>(1), Teuchos::as<LO>(1)}};
    for (int dim = 0; dim < numDimensions; ++dim) {
      localFineNodes[dim]   = lFineNodesPerDim[dim];
      localCoarseNodes[dim] = lCoarseNodesPerDim[dim];
      coarseningRate[dim]   = static_cast<LO>(
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
        (interpolationOrder == 0 || interpolationOrder == 1) &&
        (numDimensions >= 1 && numDimensions <= 3) &&
        structuredDimensionsMatch &&
        A.GetFixedBlockSize() == dofsPerNode &&
        A.getRowMap()->getLocalNumElements() == expectedFineRows &&
        Ac.getRowMap()->getLocalNumElements() == expectedCoarseRows;

    if (!useDetectedStencil)
      throw std::runtime_error(
          "StructuredRAPKernel supports only 1D, 2D, or 3D order-zero or order-one structured RAP.");

    const auto localA            = A.getLocalMatrixDevice();
    auto localAc                 = Ac.getLocalMatrixDevice();
    using local_scalar_type      = typename decltype(localA.values)::non_const_value_type;
    const local_scalar_type zero = KokkosKernels::ArithTraits<local_scalar_type>::zero();

    const size_t maxInterpolationEntries = interpolationOrder == 0
                                               ? size_t(1)
                                               : size_t(1) << numDimensions;
    const GlobalOrdinal invalidGO = Teuchos::OrdinalTraits<GlobalOrdinal>::invalid();
    const Scalar scalarZero       = Teuchos::ScalarTraits<Scalar>::zero();
    const Scalar scalarOne        = Teuchos::ScalarTraits<Scalar>::one();
    // Both interpolation orders use the compact contribution-plan implementation below.
    const size_t numFineStencilOffsets = fineStencil.stencilOffsets.size();
    Kokkos::Array<int, 3> minStencilOffset{{fineStencil.stencilOffsets[0].x,
                                            fineStencil.stencilOffsets[0].y,
                                            fineStencil.stencilOffsets[0].z}};
    Kokkos::Array<int, 3> maxStencilOffset = minStencilOffset;
    for (size_t offset = 1; offset < numFineStencilOffsets; ++offset) {
      const int values[3] = {fineStencil.stencilOffsets[offset].x,
                             fineStencil.stencilOffsets[offset].y,
                             fineStencil.stencilOffsets[offset].z};
      for (int dim = 0; dim < 3; ++dim) {
        minStencilOffset[dim] = std::min(minStencilOffset[dim], values[dim]);
        maxStencilOffset[dim] = std::max(maxStencilOffset[dim], values[dim]);
      }
    }
    const size_t numCoarseNodes =
        static_cast<size_t>(localCoarseNodes[0]) *
        static_cast<size_t>(localCoarseNodes[1]) *
        static_cast<size_t>(localCoarseNodes[2]);
    const size_t numDofs = static_cast<size_t>(dofsPerNode);

    Kokkos::Array<AxisClassMap, 3> coarseAggregatePlanClass;
    size_view reverseContributionOffsets;
    size_view reverseFineNodeOffsets;
    size_view reverseARowEntryOffsets;
    size_view reverseDirectAValueOffsets;
    scalar_view reverseCoefficients;
    Kokkos::Array<size_t, 3> numAggregatePlanClasses;
    size_t maxAcEntriesPerRow = 0;
    {
      Teuchos::TimeMonitor timer(
          *Teuchos::TimeMonitor::getNewTimer(
              timerPrefix + "prepare numeric entry plan"));

      if (interpolationOrder == 0) {
        const bool fineStencilHasCenter = std::any_of(
            fineStencil.stencilOffsets.begin(), fineStencil.stencilOffsets.end(),
            [](const auto& offset) {
              return offset.x == 0 && offset.y == 0 && offset.z == 0;
            });
        maxAcEntriesPerRow =
            (numFineStencilOffsets +
             (fineStencilHasCenter ? size_t(0) : size_t(1))) * numDofs;
      } else {
        maxAcEntriesPerRow = (numDimensions == 1 ? size_t(3)
                              : numDimensions == 2 ? size_t(9)
                                                   : size_t(27)) * numDofs;
      }

      const size_t leftBoundaryDepth[3] = {
          static_cast<size_t>(std::max(0, -minStencilOffset[0])),
          static_cast<size_t>(std::max(0, -minStencilOffset[1])),
          static_cast<size_t>(std::max(0, -minStencilOffset[2]))};
      const size_t rightBoundaryDepth[3] = {
          static_cast<size_t>(std::max(0, maxStencilOffset[0])),
          static_cast<size_t>(std::max(0, maxStencilOffset[1])),
          static_cast<size_t>(std::max(0, maxStencilOffset[2]))};

      struct AxisAggregatePlanClass {
        LO representativeCoarse;
        LO fineBegin;
        LO fineEnd;
      };
      auto buildAxisAggregatePlanClasses = [&](
                                               const LO numFine, const LO numCoarse, const LO rate,
                                               const int minOffset, const int maxOffset,
                                               const size_t leftDepth, const size_t rightDepth,
                                               std::vector<AxisAggregatePlanClass>& classes) {
        std::vector<std::vector<int>> aggregateSignatures;
        std::vector<int> aggregateSignature;
        auto aggregateClass = [&](const LO coarse) {
          const auto bounds  = interpolationOrder == 0
                                   ? aggregateBounds(coarse, numFine, numCoarse, rate)
                                   : linearSupportBounds(coarse, numFine, numCoarse, rate);
          const LO fineBegin = bounds[0], fineEnd = bounds[1];
          aggregateSignature.clear();
          aggregateSignature.push_back(static_cast<int>(
              boundaryClass(coarse, numCoarse, leftDepth, rightDepth)));
          for (LO fine = fineBegin; fine <= fineEnd; ++fine) {
            LO fineCoarse = LO(0);
            if (interpolationOrder == 0) {
              fineCoarse = constantInterpolationCoarse(fine, numFine, rate);
              aggregateSignature.push_back(static_cast<int>(boundaryClass(
                  fineCoarse, numCoarse, leftDepth, rightDepth)));
            } else {
              aggregateSignature.push_back(static_cast<int>(fine - coarseAnchor(
                  coarse, numFine, numCoarse, rate)));
            }
            for (int offset = minOffset; offset <= maxOffset; ++offset) {
              const LO neighbor = fine + static_cast<LO>(offset);
              const bool valid  = neighbor >= LO(0) && neighbor < numFine;
              aggregateSignature.push_back(valid);
              if (interpolationOrder == 0) {
                aggregateSignature.push_back(valid ? static_cast<int>(
                                                         constantInterpolationCoarse(neighbor, numFine, rate) -
                                                         fineCoarse)
                                                   : 0);
              } else if (valid) {
                LO lower = neighbor / rate;
                if (lower + 1 >= numCoarse)
                  lower = numCoarse - 2;
                const LO left  = coarseAnchor(lower, numFine, numCoarse, rate);
                const LO right = coarseAnchor(lower + 1, numFine, numCoarse, rate);
                aggregateSignature.push_back(static_cast<int>(lower - coarse));
                aggregateSignature.push_back(static_cast<int>(neighbor - left));
                aggregateSignature.push_back(static_cast<int>(right - left));
              }
            }
          }
          const auto found = std::find(
              aggregateSignatures.begin(), aggregateSignatures.end(),
              aggregateSignature);
          if (found != aggregateSignatures.end())
            return static_cast<LO>(found - aggregateSignatures.begin());
          aggregateSignatures.push_back(aggregateSignature);
          classes.push_back({coarse, fineBegin, fineEnd});
          return static_cast<LO>(classes.size() - 1);
        };

        AxisClassMap result;
        result.numCoordinates  = numCoarse;
        const LO explicitLimit = static_cast<LO>(
            2 * (leftDepth + rightDepth + size_t(3)));
        std::vector<LO> leftClasses, rightClasses;
        if (numCoarse <= explicitLimit) {
          result.leftCount = numCoarse;
          leftClasses.reserve(static_cast<size_t>(numCoarse));
          for (LO coarse = 0; coarse < numCoarse; ++coarse)
            leftClasses.push_back(aggregateClass(coarse));
        } else {
          const LO representative = numCoarse / LO(2);
          result.interiorClass    = aggregateClass(representative);
          for (LO coarse = 0; coarse < representative; ++coarse) {
            const LO id = aggregateClass(coarse);
            if (id == result.interiorClass)
              break;
            leftClasses.push_back(id);
          }
          for (LO coarse = numCoarse - 1; coarse > representative; --coarse) {
            const LO id = aggregateClass(coarse);
            if (id == result.interiorClass)
              break;
            rightClasses.push_back(id);
          }
          std::reverse(rightClasses.begin(), rightClasses.end());
          result.leftCount  = static_cast<LO>(leftClasses.size());
          result.rightCount = static_cast<LO>(rightClasses.size());
        }

        result.boundaryClasses = lo_view(
            Kokkos::ViewAllocateWithoutInitializing(
                "StructuredRAP: compact axis aggregate classes"),
            leftClasses.size() + rightClasses.size());
        auto host    = Kokkos::create_mirror_view(result.boundaryClasses);
        size_t entry = 0;
        for (const LO id : leftClasses)
          host(entry++) = id;
        for (const LO id : rightClasses)
          host(entry++) = id;
        Kokkos::deep_copy(result.boundaryClasses, host);
        return result;
      };

      std::vector<AxisAggregatePlanClass> aggregatePlanClasses[3];
      for (int dim = 0; dim < 3; ++dim) {
        coarseAggregatePlanClass[dim] = buildAxisAggregatePlanClasses(
            localFineNodes[dim], localCoarseNodes[dim], coarseningRate[dim],
            minStencilOffset[dim], maxStencilOffset[dim],
            leftBoundaryDepth[dim], rightBoundaryDepth[dim],
            aggregatePlanClasses[dim]);
        numAggregatePlanClasses[dim] = aggregatePlanClasses[dim].size();
      }

      // Only the row offsets used by representative aggregate classes are
      // needed while constructing the reusable numeric plan.  Gathering
      // those offsets avoids mirroring the complete fine and coarse row maps.
      std::vector<size_t> planARowMapIndices;
      std::vector<size_t> planAcRowMapIndices;
      for (size_t zClass = 0; zClass < numAggregatePlanClasses[2]; ++zClass) {
        const auto& zAggregate  = aggregatePlanClasses[2][zClass];
        const size_t fineZBegin = static_cast<size_t>(zAggregate.fineBegin);
        const size_t fineZEnd   = static_cast<size_t>(zAggregate.fineEnd);
        for (size_t yClass = 0; yClass < numAggregatePlanClasses[1]; ++yClass) {
          const auto& yAggregate  = aggregatePlanClasses[1][yClass];
          const size_t fineYBegin = static_cast<size_t>(yAggregate.fineBegin);
          const size_t fineYEnd   = static_cast<size_t>(yAggregate.fineEnd);
          for (size_t xClass = 0; xClass < numAggregatePlanClasses[0]; ++xClass) {
            const auto& xAggregate  = aggregatePlanClasses[0][xClass];
            const size_t fineXBegin = static_cast<size_t>(xAggregate.fineBegin);
            const size_t fineXEnd   = static_cast<size_t>(xAggregate.fineEnd);
            const size_t coarseNode =
                static_cast<size_t>(xAggregate.representativeCoarse) +
                static_cast<size_t>(localCoarseNodes[0]) *
                    (static_cast<size_t>(yAggregate.representativeCoarse) +
                     static_cast<size_t>(localCoarseNodes[1]) *
                         static_cast<size_t>(zAggregate.representativeCoarse));
            for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
              const size_t acRow = coarseNode * numDofs + rowDof;
              planAcRowMapIndices.push_back(acRow);
              planAcRowMapIndices.push_back(acRow + 1);
              for (size_t fineZ = fineZBegin; fineZ <= fineZEnd; ++fineZ) {
                for (size_t fineY = fineYBegin; fineY <= fineYEnd; ++fineY) {
                  for (size_t fineX = fineXBegin; fineX <= fineXEnd; ++fineX) {
                    const size_t fineNode =
                        fineX + static_cast<size_t>(localFineNodes[0]) *
                                    (fineY + static_cast<size_t>(localFineNodes[1]) * fineZ);
                    const size_t aRow = fineNode * numDofs + rowDof;
                    planARowMapIndices.push_back(aRow);
                    planARowMapIndices.push_back(aRow + 1);
                  }
                }
              }
            }
          }
        }
      }
      sortAndUnique(planARowMapIndices);
      sortAndUnique(planAcRowMapIndices);

      const auto gatheredARowMapHost = gatherToHost(
          localA.graph.row_map, planARowMapIndices,
          "StructuredRAP: compact A row map", executionSpace);
      const auto gatheredAcRowMapHost = gatherToHost(
          localAc.graph.row_map, planAcRowMapIndices,
          "StructuredRAP: compact Ac row map", executionSpace);
      auto aRowMapOffset = [&](const size_t rowMapIndex) {
        const size_t compactIndex = static_cast<size_t>(
            std::lower_bound(planARowMapIndices.begin(),
                             planARowMapIndices.end(), rowMapIndex) -
            planARowMapIndices.begin());
        return gatheredARowMapHost(compactIndex);
      };
      auto acRowMapOffset = [&](const size_t rowMapIndex) {
        const size_t compactIndex = static_cast<size_t>(
            std::lower_bound(planAcRowMapIndices.begin(),
                             planAcRowMapIndices.end(), rowMapIndex) -
            planAcRowMapIndices.begin());
        return gatheredAcRowMapHost(compactIndex);
      };

      const size_t reversePlanRowCount =
          numAggregatePlanClasses[0] * numAggregatePlanClasses[1] *
          numAggregatePlanClasses[2] * numDofs;
      const size_t reverseBucketCount =
          reversePlanRowCount * maxAcEntriesPerRow;
      struct PlanContribution {
        size_t fineNodeOffset, aRowEntryOffset, directAValueOffset;
        Scalar coefficient;
      };
      std::vector<std::vector<PlanContribution>> reverseBuckets(reverseBucketCount);
      struct PendingContribution {
        size_t reverseRow, fineNodeOffset, aRowEntryOffset, directAValueOffset;
        size_t acRowBegin, acRowEnd;
        size_t aEntry;
        Scalar restrictionWeight;
      };
      std::vector<PendingContribution> pendingContributions;
      std::vector<size_t> planAcEntries;
      for (size_t zClass = 0; zClass < numAggregatePlanClasses[2]; ++zClass) {
        const auto& zAggregate  = aggregatePlanClasses[2][zClass];
        const LO coarseZ        = zAggregate.representativeCoarse;
        const size_t fineZBegin = static_cast<size_t>(zAggregate.fineBegin);
        const size_t fineZEnd   = static_cast<size_t>(zAggregate.fineEnd);
        for (size_t yClass = 0; yClass < numAggregatePlanClasses[1]; ++yClass) {
          const auto& yAggregate  = aggregatePlanClasses[1][yClass];
          const LO coarseY        = yAggregate.representativeCoarse;
          const size_t fineYBegin = static_cast<size_t>(yAggregate.fineBegin);
          const size_t fineYEnd   = static_cast<size_t>(yAggregate.fineEnd);
          for (size_t xClass = 0; xClass < numAggregatePlanClasses[0]; ++xClass) {
            const auto& xAggregate  = aggregatePlanClasses[0][xClass];
            const LO coarseX        = xAggregate.representativeCoarse;
            const size_t fineXBegin = static_cast<size_t>(xAggregate.fineBegin);
            const size_t fineXEnd   = static_cast<size_t>(xAggregate.fineEnd);
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
                  ((zClass * numAggregatePlanClasses[1] + yClass) *
                       numAggregatePlanClasses[0] +
                   xClass) *
                      numDofs +
                  rowDof;
              const size_t acRow      = coarseNode * numDofs + rowDof;
              const size_t acRowBegin = acRowMapOffset(acRow);
              const size_t acRowEnd   = acRowMapOffset(acRow + 1);
              for (size_t acEntry = acRowBegin; acEntry < acRowEnd; ++acEntry)
                planAcEntries.push_back(acEntry);
              const size_t firstAValue =
                  aRowMapOffset(firstFineNode * numDofs + rowDof);
              for (size_t fineZ = fineZBegin; fineZ <= fineZEnd; ++fineZ) {
                for (size_t fineY = fineYBegin; fineY <= fineYEnd; ++fineY) {
                  for (size_t fineX = fineXBegin; fineX <= fineXEnd; ++fineX) {
                    const size_t sourceFineNode =
                        fineX + static_cast<size_t>(localFineNodes[0]) *
                                    (fineY + static_cast<size_t>(localFineNodes[1]) * fineZ);
                    const Scalar restrictionWeight = interpolationOrder == 0
                        ? scalarOne
                        : linearBasisWeight<Scalar>(
                              Kokkos::Array<LO, 3>{{static_cast<LO>(fineX),
                                                    static_cast<LO>(fineY),
                                                    static_cast<LO>(fineZ)}},
                              Kokkos::Array<LO, 3>{{coarseX, coarseY, coarseZ}},
                              localFineNodes, localCoarseNodes,
                              coarseningRate, numDimensions);
                    if (restrictionWeight == scalarZero)
                      continue;
                    const size_t aRowBegin =
                        aRowMapOffset(sourceFineNode * numDofs + rowDof);
                    const size_t aRowEnd =
                        aRowMapOffset(sourceFineNode * numDofs + rowDof + 1);
                    for (size_t aEntry = aRowBegin; aEntry < aRowEnd; ++aEntry)
                      pendingContributions.push_back(PendingContribution{
                          reverseRow, sourceFineNode - firstFineNode,
                          aEntry - aRowBegin, aEntry - firstAValue,
                          acRowBegin, acRowEnd,
                          aEntry, restrictionWeight});
                  }
                }
              }
            }
          }
        }
      }

      std::vector<size_t> planAEntries;
      planAEntries.reserve(pendingContributions.size());
      for (const auto& contribution : pendingContributions)
        planAEntries.push_back(contribution.aEntry);
      sortAndUnique(planAEntries);
      sortAndUnique(planAcEntries);
      const auto gatheredAEntriesHost = gatherToHost(
          localA.graph.entries, planAEntries,
          "StructuredRAP: compact A entries", executionSpace);
      const auto gatheredAcEntriesHost = gatherToHost(
          localAc.graph.entries, planAcEntries,
          "StructuredRAP: compact Ac entries", executionSpace);

      std::vector<LO> planFineColumns;
      planFineColumns.reserve(pendingContributions.size());
      for (const auto& contribution : pendingContributions) {
        const size_t fineEntry = static_cast<size_t>(std::lower_bound(
            planAEntries.begin(), planAEntries.end(), contribution.aEntry) -
            planAEntries.begin());
        planFineColumns.push_back(gatheredAEntriesHost(fineEntry));
      }
      std::sort(planFineColumns.begin(), planFineColumns.end());
      planFineColumns.erase(
          std::unique(planFineColumns.begin(), planFineColumns.end()),
          planFineColumns.end());

      using TargetArray = Kokkos::Array<GlobalOrdinal, 8>;
      using WeightArray = Kokkos::Array<Scalar, 3>;
      using TargetWeightArray = Kokkos::Array<Scalar, 8>;
      std::vector<TargetArray> planTargetGids(planFineColumns.size());
      std::vector<TargetWeightArray> planTargetWeights(planFineColumns.size());
      const auto fineColumnMap = A.getColMap();
      const auto fineRowMap    = A.getRowMap();
      const auto coarseRowMap  = Ac.getRowMap();
      const auto coarseColumnMap = Ac.getColMap();
      const LO invalidLO = Teuchos::OrdinalTraits<LO>::invalid();

      auto fillInterpolationMetadata =
          [&](const LO fineRow, TargetArray& targets,
              TargetWeightArray& targetWeights) {
            for (size_t entry = 0; entry < size_t(8); ++entry) {
              targets[entry] = invalidGO;
              targetWeights[entry] = scalarZero;
            }
            WeightArray axisWeights;
            for (int dim = 0; dim < 3; ++dim)
              axisWeights[dim] = scalarZero;

            const LO fineNode = fineRow / dofsPerNode;
            const LO dof      = fineRow % dofsPerNode;
            const LO fine[3] = {
                fineNode % localFineNodes[0],
                (fineNode / localFineNodes[0]) % localFineNodes[1],
                fineNode / (localFineNodes[0] * localFineNodes[1])};
            LO coarseBase[3] = {LO(0), LO(0), LO(0)};
            for (int dim = 0; dim < numDimensions; ++dim) {
              if (interpolationOrder == 0) {
                coarseBase[dim] = constantInterpolationCoarse(
                    fine[dim], localFineNodes[dim], coarseningRate[dim]);
              } else {
                LO lower = fine[dim] / coarseningRate[dim];
                if (lower + 1 >= localCoarseNodes[dim])
                  lower = localCoarseNodes[dim] - 2;
                coarseBase[dim] = lower;
                const LO left = coarseAnchor(
                    lower, localFineNodes[dim], localCoarseNodes[dim], coarseningRate[dim]);
                const LO right = coarseAnchor(
                    lower + 1, localFineNodes[dim], localCoarseNodes[dim], coarseningRate[dim]);
                axisWeights[dim] =
                    Scalar(fine[dim] - left) / Scalar(right - left);
              }
            }

            size_t target = 0;
            for (size_t corner = 0; corner < maxInterpolationEntries; ++corner) {
              Scalar weight = scalarOne;
              LO coarse[3] = {coarseBase[0], coarseBase[1], coarseBase[2]};
              for (int dim = 0; dim < numDimensions; ++dim) {
                const bool upper = interpolationOrder == 1 &&
                                   (corner & (size_t(1) << dim));
                if (upper)
                  ++coarse[dim];
                if (interpolationOrder == 1)
                  weight *= upper ? axisWeights[dim]
                                  : scalarOne - axisWeights[dim];
              }
              if (weight == scalarZero)
                continue;
              const size_t coarseNode =
                  static_cast<size_t>(coarse[0]) +
                  static_cast<size_t>(localCoarseNodes[0]) *
                      (static_cast<size_t>(coarse[1]) +
                       static_cast<size_t>(localCoarseNodes[1]) * coarse[2]);
              targets[target++] = coarseRowMap->getGlobalElement(
                  static_cast<LO>(coarseNode * numDofs + dof));
              targetWeights[target - 1] = weight;
            }
          };

      Teuchos::Array<GlobalOrdinal> remoteFineGids;
      std::vector<size_t> remotePlanColumns;
      for (size_t column = 0; column < planFineColumns.size(); ++column) {
        const GlobalOrdinal fineGid =
            fineColumnMap->getGlobalElement(planFineColumns[column]);
        const LO ownedFine = fineRowMap->getLocalElement(fineGid);
        if (ownedFine != invalidLO) {
          fillInterpolationMetadata(
              ownedFine, planTargetGids[column], planTargetWeights[column]);
        } else {
          remoteFineGids.push_back(fineGid);
          remotePlanColumns.push_back(column);
        }
      }

      if (distributed) {
        const auto remoteFineMap =
            Xpetra::MapFactory<LO, GlobalOrdinal, Node>::Build(
                A.getRowMap()->lib(),
                Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid(),
                remoteFineGids(), A.getRowMap()->getIndexBase(),
                A.getRowMap()->getComm());
        const auto remoteImporter =
            Xpetra::ImportFactory<LO, GlobalOrdinal, Node>::Build(
                A.getRowMap(), remoteFineMap);

        auto ownedTargets =
            Xpetra::MultiVectorFactory<GlobalOrdinal, LO, GlobalOrdinal, Node>::Build(
                A.getRowMap(), maxInterpolationEntries, false);
        auto ownedTargetWeights =
            Xpetra::MultiVectorFactory<Scalar, LO, GlobalOrdinal, Node>::Build(
                A.getRowMap(), maxInterpolationEntries, false);
        auto ownedTargetData =
            ownedTargets->getLocalViewHost(Tpetra::Access::OverwriteAll);
        auto ownedTargetWeightData =
            ownedTargetWeights->getLocalViewHost(Tpetra::Access::OverwriteAll);
        const auto exportLids = remoteImporter->getExportLIDs();
        for (const LO exportLid : exportLids) {
          TargetArray targets;
          TargetWeightArray targetWeights;
          fillInterpolationMetadata(exportLid, targets, targetWeights);
          for (size_t entry = 0; entry < maxInterpolationEntries; ++entry) {
            ownedTargetData(exportLid, entry) = targets[entry];
            ownedTargetWeightData(exportLid, entry) = targetWeights[entry];
          }
        }

        auto remoteTargets =
            Xpetra::MultiVectorFactory<GlobalOrdinal, LO, GlobalOrdinal, Node>::Build(
                remoteFineMap, maxInterpolationEntries, false);
        auto remoteTargetWeights =
            Xpetra::MultiVectorFactory<Scalar, LO, GlobalOrdinal, Node>::Build(
                remoteFineMap, maxInterpolationEntries, false);
        remoteTargets->doImport(*ownedTargets, *remoteImporter, Xpetra::INSERT);
        remoteTargetWeights->doImport(
            *ownedTargetWeights, *remoteImporter, Xpetra::INSERT);
        const auto remoteTargetData =
            remoteTargets->getLocalViewHost(Tpetra::Access::ReadOnly);
        const auto remoteTargetWeightData =
            remoteTargetWeights->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t remote = 0; remote < remotePlanColumns.size(); ++remote) {
          const size_t column = remotePlanColumns[remote];
          for (size_t entry = 0; entry < maxInterpolationEntries; ++entry) {
            planTargetGids[column][entry] = remoteTargetData(remote, entry);
            planTargetWeights[column][entry] =
                remoteTargetWeightData(remote, entry);
          }
        }
      }

      for (const auto& contribution : pendingContributions) {
        const size_t fineEntry = static_cast<size_t>(std::lower_bound(
            planAEntries.begin(), planAEntries.end(), contribution.aEntry) -
            planAEntries.begin());
        const LO fineColumn = gatheredAEntriesHost(fineEntry);
        const size_t column = static_cast<size_t>(std::lower_bound(
            planFineColumns.begin(), planFineColumns.end(), fineColumn) -
            planFineColumns.begin());
        for (size_t interpolation = 0;
             interpolation < maxInterpolationEntries; ++interpolation) {
          const GlobalOrdinal targetGid =
              planTargetGids[column][interpolation];
          if (targetGid == invalidGO)
            continue;
          const Scalar targetWeight = planTargetWeights[column][interpolation];
          const LO targetColumn =
              coarseColumnMap->getLocalElement(targetGid);
          if (targetColumn == invalidLO)
            continue;
          for (size_t acEntry = contribution.acRowBegin;
               acEntry < contribution.acRowEnd; ++acEntry) {
            const size_t compactAcEntry = static_cast<size_t>(
                std::lower_bound(planAcEntries.begin(), planAcEntries.end(),
                                 acEntry) -
                planAcEntries.begin());
            if (gatheredAcEntriesHost(compactAcEntry) == targetColumn) {
              const size_t bucket =
                  contribution.reverseRow * maxAcEntriesPerRow +
                  acEntry - contribution.acRowBegin;
              reverseBuckets[bucket].push_back(PlanContribution{
                  contribution.fineNodeOffset,
                  contribution.aRowEntryOffset,
                  contribution.directAValueOffset,
                  contribution.restrictionWeight * targetWeight});
              break;
            }
          }
        }
      }
      size_t numContributions = 0;
      for (const auto& bucket : reverseBuckets)
        numContributions += bucket.size();
      PlanArray<size_view> offsets(
          "StructuredRAP: reverse contribution offsets",
          reverseBucketCount + 1);
      PlanArray<size_view> fineNodeOffsets(
          "StructuredRAP: reverse fine node offsets", numContributions);
      PlanArray<size_view> aRowEntryOffsets(
          "StructuredRAP: reverse A row entry offsets", numContributions);
      PlanArray<size_view> directAValueOffsets(
          "StructuredRAP: reverse direct A value offsets", numContributions);
      size_t contribution = 0;
      for (size_t bucket = 0; bucket < reverseBucketCount; ++bucket) {
        offsets.host(bucket) = contribution;
        for (const auto& entry : reverseBuckets[bucket]) {
          fineNodeOffsets.host(contribution)     = entry.fineNodeOffset;
          aRowEntryOffsets.host(contribution)    = entry.aRowEntryOffset;
          directAValueOffsets.host(contribution) = entry.directAValueOffset;
          ++contribution;
        }
      }
      offsets.host(reverseBucketCount) = contribution;
      reverseContributionOffsets       = offsets.copyToDevice(executionSpace);
      reverseFineNodeOffsets           = fineNodeOffsets.copyToDevice(executionSpace);
      reverseARowEntryOffsets          = aRowEntryOffsets.copyToDevice(executionSpace);
      reverseDirectAValueOffsets       = directAValueOffsets.copyToDevice(executionSpace);
      if (interpolationOrder == 1) {
        PlanArray<scalar_view> coefficients(
            "StructuredRAP: reverse interpolation coefficients", numContributions);
        contribution = 0;
        for (const auto& bucket : reverseBuckets)
          for (const auto& entry : bucket)
            coefficients.host(contribution++) = entry.coefficient;
        reverseCoefficients = coefficients.copyToDevice(executionSpace);
        executionSpace.fence("StructuredRAP: interpolation coefficients ready");
      }

      executionSpace.fence("StructuredRAP: numeric entry plan prepared");
    }

    {
      Teuchos::TimeMonitor timer(
          *Teuchos::TimeMonitor::getNewTimer(
              timerPrefix +
              "numeric (structured interpolation)"));
      auto runNumeric = [&](auto interpolationOrderTag) {
        constexpr int numericInterpolationOrder =
            decltype(interpolationOrderTag)::value;
        range_policy policy(executionSpace, 0, numCoarseNodes);
        Kokkos::parallel_for(
            "StructuredRAP: structured interpolation P^T A P",
            policy,
            KOKKOS_LAMBDA(const size_t coarseNodeIndex) {
            const LO coarseNode         = static_cast<LO>(coarseNodeIndex);
            const LO coarseX            = coarseNode % localCoarseNodes[0];
            const LO coarseYZ           = coarseNode / localCoarseNodes[0];
            const LO coarseY            = coarseYZ % localCoarseNodes[1];
            const LO coarseZ            = coarseYZ / localCoarseNodes[1];
            const size_t firstCoarseRow = coarseNodeIndex * numDofs;

            const auto fineXBounds = numericInterpolationOrder == 0
                ? aggregateBounds(coarseX, localFineNodes[0], localCoarseNodes[0], coarseningRate[0])
                : linearSupportBounds(coarseX, localFineNodes[0], localCoarseNodes[0], coarseningRate[0]);
            const auto fineYBounds = numericInterpolationOrder == 0
                ? aggregateBounds(coarseY, localFineNodes[1], localCoarseNodes[1], coarseningRate[1])
                : linearSupportBounds(coarseY, localFineNodes[1], localCoarseNodes[1], coarseningRate[1]);
            const auto fineZBounds = numericInterpolationOrder == 0
                ? aggregateBounds(coarseZ, localFineNodes[2], localCoarseNodes[2], coarseningRate[2])
                : linearSupportBounds(coarseZ, localFineNodes[2], localCoarseNodes[2], coarseningRate[2]);
            const size_t firstFineNode =
                static_cast<size_t>(fineXBounds[0]) +
                static_cast<size_t>(localFineNodes[0]) *
                    (static_cast<size_t>(fineYBounds[0]) +
                     static_cast<size_t>(localFineNodes[1]) * fineZBounds[0]);
            const bool useDirectOffsets =
                coarseAggregatePlanClass[0].isInterior(coarseX) &&
                (numDimensions < 2 ||
                 coarseAggregatePlanClass[1].isInterior(coarseY)) &&
                (numDimensions < 3 ||
                 coarseAggregatePlanClass[2].isInterior(coarseZ));
            const size_t reversePlanNodeBegin =
                ((static_cast<size_t>(coarseAggregatePlanClass[2](coarseZ)) *
                      numAggregatePlanClasses[1] +
                  coarseAggregatePlanClass[1](coarseY)) *
                     numAggregatePlanClasses[0] +
                 coarseAggregatePlanClass[0](coarseX)) *
                numDofs;

            for (size_t rowDof = 0; rowDof < numDofs; ++rowDof) {
              const size_t acRow          = firstCoarseRow + rowDof;
              const size_t acRowBegin     = localAc.graph.row_map(acRow);
              const size_t acRowEnd       = localAc.graph.row_map(acRow + size_t(1));
              const size_t reversePlanRow = reversePlanNodeBegin + rowDof;
              const size_t firstAValue =
                  localA.graph.row_map(firstFineNode * numDofs + rowDof);
              for (size_t acOffset = 0; acOffset < acRowEnd - acRowBegin; ++acOffset) {
                const size_t bucket =
                    reversePlanRow * maxAcEntriesPerRow + acOffset;
                const size_t contributionBegin = reverseContributionOffsets(bucket);
                const size_t contributionEnd   = reverseContributionOffsets(bucket + 1);
                local_scalar_type sum0 = zero, sum1 = zero, sum2 = zero, sum3 = zero;
                size_t contribution = contributionBegin;
                if (useDirectOffsets) {
                  for (; contribution + 3 < contributionEnd; contribution += 4) {
                    sum0 += weightedAValue<numericInterpolationOrder>(
                        localA, firstAValue + reverseDirectAValueOffsets(contribution),
                        reverseCoefficients, contribution);
                    sum1 += weightedAValue<numericInterpolationOrder>(
                        localA, firstAValue + reverseDirectAValueOffsets(contribution + 1),
                        reverseCoefficients, contribution + 1);
                    sum2 += weightedAValue<numericInterpolationOrder>(
                        localA, firstAValue + reverseDirectAValueOffsets(contribution + 2),
                        reverseCoefficients, contribution + 2);
                    sum3 += weightedAValue<numericInterpolationOrder>(
                        localA, firstAValue + reverseDirectAValueOffsets(contribution + 3),
                        reverseCoefficients, contribution + 3);
                  }
                  local_scalar_type sum = (sum0 + sum1) + (sum2 + sum3);
                  for (; contribution < contributionEnd; ++contribution)
                    sum += weightedAValue<numericInterpolationOrder>(
                        localA, firstAValue + reverseDirectAValueOffsets(contribution),
                        reverseCoefficients, contribution);
                  localAc.values(acRowBegin + acOffset) = sum;
                } else {
                  for (; contribution + 3 < contributionEnd; contribution += 4) {
                    const size_t aRow0 =
                        (firstFineNode + reverseFineNodeOffsets(contribution)) *
                            numDofs +
                        rowDof;
                    const size_t aRow1 =
                        (firstFineNode + reverseFineNodeOffsets(contribution + 1)) *
                            numDofs +
                        rowDof;
                    const size_t aRow2 =
                        (firstFineNode + reverseFineNodeOffsets(contribution + 2)) *
                            numDofs +
                        rowDof;
                    const size_t aRow3 =
                        (firstFineNode + reverseFineNodeOffsets(contribution + 3)) *
                            numDofs +
                        rowDof;
                    sum0 += weightedAValue<numericInterpolationOrder>(
                        localA, localA.graph.row_map(aRow0) + reverseARowEntryOffsets(contribution),
                        reverseCoefficients, contribution);
                    sum1 += weightedAValue<numericInterpolationOrder>(
                        localA, localA.graph.row_map(aRow1) + reverseARowEntryOffsets(contribution + 1),
                        reverseCoefficients, contribution + 1);
                    sum2 += weightedAValue<numericInterpolationOrder>(
                        localA, localA.graph.row_map(aRow2) + reverseARowEntryOffsets(contribution + 2),
                        reverseCoefficients, contribution + 2);
                    sum3 += weightedAValue<numericInterpolationOrder>(
                        localA, localA.graph.row_map(aRow3) + reverseARowEntryOffsets(contribution + 3),
                        reverseCoefficients, contribution + 3);
                  }
                  local_scalar_type sum = (sum0 + sum1) + (sum2 + sum3);
                  for (; contribution < contributionEnd; ++contribution) {
                    const size_t aRow =
                        (firstFineNode + reverseFineNodeOffsets(contribution)) *
                            numDofs +
                        rowDof;
                    sum += weightedAValue<numericInterpolationOrder>(
                        localA, localA.graph.row_map(aRow) + reverseARowEntryOffsets(contribution),
                        reverseCoefficients, contribution);
                  }
                  localAc.values(acRowBegin + acOffset) = sum;
                }
              }
            }
            });
        executionSpace.fence(
            "StructuredRAP: structured product complete");
      };
      if (interpolationOrder == 0)
        runNumeric(std::integral_constant<int, 0>{});
      else
        runNumeric(std::integral_constant<int, 1>{});
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
