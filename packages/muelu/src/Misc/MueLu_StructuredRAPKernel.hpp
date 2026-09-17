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

#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_TestForException.hpp>

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
 * kernel is graph driven: no assumption about a particular stencil or
 * interpolation order is made here.  On multiple ranks, contributions from
 * locally owned fine rows are accumulated into overlapping coarse rows and
 * then exported to the owning ranks.
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
  using device_type = typename Node::device_type;
  using size_view   = Kokkos::View<size_t*, device_type>;
  using lo_view     = Kokkos::View<LocalOrdinal*, device_type>;

  struct GraphCache {
    Teuchos::RCP<const Graph> aGraph;
    Teuchos::RCP<const Graph> pGraph;
    Teuchos::RCP<const Graph> acGraph;
    Teuchos::RCP<Matrix> importedP;
    Teuchos::RCP<Matrix> overlapAc;
    size_view transposeRowPtr;
    lo_view transposeFineRows;
    size_view transposePEntries;
    lo_view ghostedPToAcCol;
    bool pToAcColumnMapIsIdentity = false;
    size_t maxAcRowEntries        = 0;
  };

  static constexpr const char* cacheKey() {
    return "structured rap: graph cache";
  }

 public:
  static void Compute(const Matrix& A,
                      const Matrix& P,
                      Matrix& Ac,
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

    const std::string prefix = "MueLu::Details::StructuredRAPKernel::Compute: ";

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

    const Teuchos::RCP<const Graph> aGraph  = A.getCrsGraph();
    const Teuchos::RCP<const Graph> pGraph  = P.getCrsGraph();
    const Teuchos::RCP<const Graph> acGraph = Ac.getCrsGraph();
    Teuchos::RCP<GraphCache> cache;
    if (!params.is_null() && params->isType<Teuchos::RCP<GraphCache>>(cacheKey())) {
      cache = params->get<Teuchos::RCP<GraphCache>>(cacheKey());
      params->remove(cacheKey());
    }

    const bool cacheValid = !cache.is_null() &&
                            cache->aGraph.getRawPtr() == aGraph.getRawPtr() &&
                            cache->pGraph.getRawPtr() == pGraph.getRawPtr() &&
                            cache->acGraph.getRawPtr() == acGraph.getRawPtr();
    if (!cacheValid) {
      cache          = Teuchos::rcp(new GraphCache);
      cache->aGraph  = aGraph;
      cache->pGraph  = pGraph;
      cache->acGraph = acGraph;
    }

    const Teuchos::RCP<const Matrix> Pptr = Teuchos::rcpFromRef(P);

    // A's column map identifies every P row needed by the local A rows.  Import
    // those rows before the device kernel so all numeric traversal is local.
    const Teuchos::RCP<const Import> fineRowImporter = A.getCrsGraph()->getImporter();
    Teuchos::RCP<const Matrix> ghostedP              = Pptr;
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

    // A local fine row can contribute to coarse rows owned by another rank.
    // Import the supplied Ac graph onto P's column map, compute those overlap
    // rows, and sum them back to the owning Ac rows after the device kernel.
    const Teuchos::RCP<const Import> coarseRowImporter = P.getCrsGraph()->getImporter();
    Teuchos::RCP<Matrix> overlapAc                     = Teuchos::rcpFromRef(Ac);
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

    const SC zero = Teuchos::ScalarTraits<SC>::zero();
    Ac.setAllToScalar(zero);
    if (overlapAc.getRawPtr() != &Ac)
      overlapAc->setAllToScalar(zero);

    const auto localA         = A.getLocalMatrixDevice();
    const auto localP         = P.getLocalMatrixDevice();
    const auto localGhostedP  = ghostedP->getLocalMatrixDevice();
    auto localOverlapAc       = overlapAc->getLocalMatrixDevice();
    const size_t numFineRows  = localA.numRows();
    const size_t numCoarseRows = localOverlapAc.numRows();

    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(localP.numRows()) != numFineRows,
                               std::runtime_error,
                               prefix << "A and P have different local row counts.");
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(localGhostedP.numRows()) != A.getColMap()->getLocalNumElements(),
                               std::runtime_error,
                               prefix << "the imported P row map does not match A's column map.");
    TEUCHOS_TEST_FOR_EXCEPTION(numCoarseRows != P.getColMap()->getLocalNumElements(),
                               std::runtime_error,
                               prefix << "the overlap Ac row map does not match P's column map.");

    execution_space executionSpace;
    const bool serialExecution = executionSpace.concurrency() == 1;

    size_view transposeRowPtr;
    lo_view transposeFineRows;
    size_view transposePEntries;
    lo_view ghostedPToAcCol;
    bool pToAcColumnMapIsIdentity = false;
    const LO invalidLO          = Teuchos::OrdinalTraits<LO>::invalid();

    Kokkos::View<int, device_type> missingGraphEntry("StructuredRAP: missing graph entry");
    Kokkos::deep_copy(executionSpace, missingGraphEntry, 0);

    if (!cacheValid) {
      if (!serialExecution) {
        // Form a lightweight local transpose adjacency for P.  Values remain in P;
        // only fine-row and P-entry indices are stored here.
        size_view transposeCounts("StructuredRAP: transpose counts", numCoarseRows);
        Kokkos::deep_copy(executionSpace, transposeCounts, size_t(0));
        Kokkos::parallel_for(
            "StructuredRAP: count transpose entries", range_policy(executionSpace, 0, numFineRows),
            KOKKOS_LAMBDA(const size_t fineRow) {
              const auto pRow = localP.rowConst(static_cast<LO>(fineRow));
              for (LO entry = 0; entry < pRow.length; ++entry)
                Kokkos::atomic_fetch_add(&transposeCounts(static_cast<size_t>(pRow.colidx(entry))), size_t(1));
            });

        transposeRowPtr = size_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose row pointers"), numCoarseRows + 1);
        Kokkos::parallel_scan(
            "StructuredRAP: scan transpose rows", range_policy(executionSpace, 0, numCoarseRows + 1),
            KOKKOS_LAMBDA(const size_t row, size_t& update, const bool final) {
              if (final)
                transposeRowPtr(row) = update;
              if (row < numCoarseRows)
                update += transposeCounts(row);
            });

        const size_t localPEntries = localP.nnz();
        transposeFineRows = lo_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose fine rows"), localPEntries);
        transposePEntries = size_view(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose P entries"), localPEntries);
        size_view transposeOffsets(
            Kokkos::ViewAllocateWithoutInitializing("StructuredRAP: transpose offsets"), numCoarseRows);
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

      // The imported P and overlapping Ac may use different local column-map
      // orderings.  Translate once outside the product traversal.
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

    if constexpr (executionSpaceIsHost) {
      if (serialExecution) {
        Kokkos::parallel_for(
            "StructuredRAP: serial fine-row P^T A P", range_policy(executionSpace, 0, numFineRows),
            KOKKOS_LAMBDA(const size_t fineRow) {
              const auto leftPRow = localP.rowConst(static_cast<LO>(fineRow));
              const auto aRow     = localA.rowConst(static_cast<LO>(fineRow));
              for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                const SC aValue = aRow.value(aEntry);
                const auto rightPRow = localGhostedP.rowConst(aRow.colidx(aEntry));
                for (LO leftEntry = 0; leftEntry < leftPRow.length; ++leftEntry) {
                  const size_t coarseRow = static_cast<size_t>(leftPRow.colidx(leftEntry));
                  const SC raValue       = leftPRow.value(leftEntry) * aValue;
                  const size_t acBegin   = localOverlapAc.graph.row_map(coarseRow);
                  const size_t acEnd     = localOverlapAc.graph.row_map(coarseRow + 1);
                  for (LO rightEntry = 0; rightEntry < rightPRow.length; ++rightEntry) {
                    const LO pColumn = rightPRow.colidx(rightEntry);
                    const LO acColumn = pToAcColumnMapIsIdentity
                                            ? pColumn
                                            : ghostedPToAcCol(static_cast<size_t>(pColumn));
                    if (acColumn == invalidLO) {
                      Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                      continue;
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
                      continue;
                    }
                    localOverlapAc.values(lower) += raValue * rightPRow.value(rightEntry);
                  }
                }
              }
            });
      } else {
        Kokkos::parallel_for(
            "StructuredRAP: CPU row P^T A P", range_policy(executionSpace, 0, numCoarseRows),
            KOKKOS_LAMBDA(const size_t coarseRow) {
              const size_t acBegin = localOverlapAc.graph.row_map(coarseRow);
              const size_t acEnd   = localOverlapAc.graph.row_map(coarseRow + 1);
              const size_t begin   = transposeRowPtr(coarseRow);
              const size_t end     = transposeRowPtr(coarseRow + 1);
              for (size_t transposeEntry = begin; transposeEntry < end; ++transposeEntry) {
                const LO fineRow     = transposeFineRows(transposeEntry);
                const SC restriction = localP.values(transposePEntries(transposeEntry));
                const auto aRow      = localA.rowConst(fineRow);

                for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                  const SC raValue = restriction * aRow.value(aEntry);
                  const auto pRow  = localGhostedP.rowConst(aRow.colidx(aEntry));
                  for (LO pEntry = 0; pEntry < pRow.length; ++pEntry) {
                    const LO pColumn = pRow.colidx(pEntry);
                    const LO acColumn = pToAcColumnMapIsIdentity
                                            ? pColumn
                                            : ghostedPToAcCol(static_cast<size_t>(pColumn));
                    if (acColumn == invalidLO) {
                      Kokkos::atomic_exchange(&missingGraphEntry(), 1);
                      continue;
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
                      continue;
                    }
                    localOverlapAc.values(lower) += raValue * pRow.value(pEntry);
                  }
                }
              }
            });
      }
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
              const LO fineRow     = transposeFineRows(transposeEntry);
              const SC restriction = localP.values(transposePEntries(transposeEntry));
              const auto aRow      = localA.rowConst(fineRow);

              for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                const LO ghostedPRow = aRow.colidx(aEntry);
                const SC raValue     = restriction * aRow.value(aEntry);
                const auto pRow      = localGhostedP.rowConst(ghostedPRow);

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, pRow.length),
                    [&](const LO pEntry) {
                      const LO pColumn = pRow.colidx(pEntry);
                      const LO acColumn = pToAcColumnMapIsIdentity
                                              ? pColumn
                                              : ghostedPToAcCol(static_cast<size_t>(pColumn));
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
                      localOverlapAc.values(lower) += raValue * pRow.value(pEntry);
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
                    const LO fineRow       = transposeFineRows(transposeEntry);
                    const SC restriction   = localP.values(transposePEntries(transposeEntry));
                    const auto aRow        = localA.rowConst(fineRow);

                    for (LO aEntry = 0; aEntry < aRow.length; ++aEntry) {
                      const LO ghostedPRow = aRow.colidx(aEntry);
                      const SC raValue     = restriction * aRow.value(aEntry);
                      const auto pRow      = localGhostedP.rowConst(ghostedPRow);

                      Kokkos::parallel_for(
                          Kokkos::ThreadVectorRange(team, pRow.length),
                          [&](const LO pEntry) {
                            const LO pColumn = pRow.colidx(pEntry);
                            const LO acColumn = pToAcColumnMapIsIdentity
                                                    ? pColumn
                                                    : ghostedPToAcCol(static_cast<size_t>(pColumn));
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

                            Kokkos::atomic_add(&rowValues(lower - acBegin), raValue * pRow.value(pEntry));
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

    int missingGraphEntryHost = 0;
    Kokkos::deep_copy(executionSpace, missingGraphEntryHost, missingGraphEntry);
    TEUCHOS_TEST_FOR_EXCEPTION(missingGraphEntryHost != 0,
                               std::runtime_error,
                               prefix << "the supplied Ac graph does not contain every entry generated by P^T A P.");

    if (!coarseRowImporter.is_null()) {
      Ac.doExport(*overlapAc, *coarseRowImporter, Xpetra::ADD);
    }

    // MatrixFactory::Build(graph) creates a fill-active matrix even though its
    // static graph is already fill complete.  Match the state produced by the
    // Xpetra triple-product path before returning it to StructuredRAPFactory.
    if (!Ac.isFillComplete())
      Ac.fillComplete(P.getDomainMap(), P.getDomainMap(), params);

    if (!params.is_null())
      params->set(cacheKey(), cache);
  }
};

}  // namespace Details
}  // namespace MueLu

#endif  // MUELU_STRUCTUREDRAPKERNEL_HPP
