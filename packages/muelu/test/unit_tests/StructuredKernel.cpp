// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_ScalarTraits.hpp>

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "MueLu_config.hpp"

#include "MueLu_TestHelpers.hpp"
#include "MueLu_Version.hpp"

#include <Galeri_XpetraMaps.hpp>

#include <Xpetra_MapFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_TripleMatrixMultiply.hpp>

#include "MueLu_AmalgamationFactory.hpp"
#include "MueLu_CoalesceDropFactory.hpp"
#include "MueLu_GeometricInterpolationPFactory.hpp"
#include "MueLu_NoFactory.hpp"
#include "MueLu_StructuredAggregationFactory.hpp"
#include "MueLu_StructuredRAPFactory.hpp"
#include "MueLu_StructuredRAPKernel.hpp"

namespace MueLuTests {

namespace {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
struct Elasticity2DData {
  using Matrix = Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  Teuchos::RCP<Matrix> A;
  Teuchos::RCP<Matrix> P;
  Teuchos::Array<LocalOrdinal> fineNodesPerDim;
  Teuchos::Array<LocalOrdinal> coarseNodesPerDim;
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Elasticity2DData<Scalar, LocalOrdinal, GlobalOrdinal, Node>
buildElasticity2DData(const GlobalOrdinal nx,
                      const GlobalOrdinal ny,
                      const int coarseningRate) {
  using SC                    = Scalar;
  using LO                    = LocalOrdinal;
  using GO                    = GlobalOrdinal;
  using NO                    = Node;
  using Map                   = Xpetra::Map<LO, GO, NO>;
  using Matrix                = Xpetra::Matrix<SC, LO, GO, NO>;
  using CrsMatrixWrap         = Xpetra::CrsMatrixWrap<SC, LO, GO, NO>;
  using MultiVector           = Xpetra::MultiVector<SC, LO, GO, NO>;
  using coordinate_type       = typename Teuchos::ScalarTraits<SC>::coordinateType;
  using RealValuedMultiVector = Xpetra::MultiVector<coordinate_type, LO, GO, NO>;
  using AmalgamationFactory   = MueLu::AmalgamationFactory<SC, LO, GO, NO>;
  using CoalesceDropFactory   = MueLu::CoalesceDropFactory<SC, LO, GO, NO>;
  using AggregationFactory    = MueLu::StructuredAggregationFactory<SC, LO, GO, NO>;
  using ProlongatorFactory    = MueLu::GeometricInterpolationPFactory<SC, LO, GO, NO>;

  Teuchos::RCP<const Teuchos::Comm<int> > comm =
      TestHelpers::Parameters::getDefaultComm();

  Teuchos::ParameterList galeriList;
  galeriList.set("nx", nx);
  galeriList.set("ny", ny);

  Teuchos::RCP<const Map> nodalMap =
      Galeri::Xpetra::CreateMap<LO, GO, NO>(
          TestHelpers::Parameters::getLib(), "Cartesian2D", comm,
          galeriList);
  Teuchos::RCP<RealValuedMultiVector> coordinates =
      Galeri::Xpetra::Utils::CreateCartesianCoordinates<
          coordinate_type, LO, GO, Map, RealValuedMultiVector>(
          "2D", nodalMap, galeriList);
  Teuchos::RCP<const Map> dofMap =
      Xpetra::MapFactory<LO, GO, NO>::Build(nodalMap, 2);

  Teuchos::RCP<Galeri::Xpetra::Problem<Map, CrsMatrixWrap, MultiVector> >
      galeriProblem =
          Galeri::Xpetra::BuildProblem<SC, LO, GO, Map, CrsMatrixWrap,
                                       MultiVector>(
              "Elasticity2D", dofMap, galeriList);

  Elasticity2DData<SC, LO, GO, NO> data;
  data.A = galeriProblem->BuildMatrix();
  data.A->SetFixedBlockSize(2);
  data.fineNodesPerDim = Teuchos::Array<LO>(3, Teuchos::as<LO>(1));
  data.fineNodesPerDim[0] = galeriList.get<LO>("lnx");
  data.fineNodesPerDim[1] = galeriList.get<LO>("lny");

  MueLu::Level fineLevel, coarseLevel;
  TestHelpers::TestFactory<SC, LO, GO, NO>::createTwoLevelHierarchy(
      fineLevel, coarseLevel);
  fineLevel.Set("A", data.A);
  fineLevel.Set("Coordinates", coordinates);
  fineLevel.Set("numDimensions", 2);
  fineLevel.Set("lNodesPerDim", data.fineNodesPerDim);

  Teuchos::RCP<MultiVector> nullspace =
      Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(
          data.A->getRowMap(), 1);
  nullspace->putScalar(Teuchos::ScalarTraits<SC>::one());
  fineLevel.Set("Nullspace", nullspace);

  Teuchos::RCP<AmalgamationFactory> amalgamation =
      Teuchos::rcp(new AmalgamationFactory());
  Teuchos::RCP<CoalesceDropFactory> coalesceDrop =
      Teuchos::rcp(new CoalesceDropFactory());
  coalesceDrop->SetFactory("UnAmalgamationInfo", amalgamation);

  Teuchos::RCP<AggregationFactory> aggregation =
      Teuchos::rcp(new AggregationFactory());
  aggregation->SetParameter(
      "aggregation: mode", Teuchos::ParameterEntry(std::string("uncoupled")));
  aggregation->SetParameter(
      "aggregation: output type",
      Teuchos::ParameterEntry(std::string("CrsGraph")));
  aggregation->SetParameter(
      "aggregation: coarsening order", Teuchos::ParameterEntry(0));
  aggregation->SetParameter(
      "aggregation: coarsening rate",
      Teuchos::ParameterEntry(std::string("{") +
                              std::to_string(coarseningRate) + "}"));
  aggregation->SetFactory("Graph", coalesceDrop);
  aggregation->SetFactory("DofsPerNode", coalesceDrop);

  Teuchos::RCP<ProlongatorFactory> prolongator =
      Teuchos::rcp(new ProlongatorFactory());
  prolongator->SetFactory("A", MueLu::NoFactory::getRCP());
  prolongator->SetFactory("Coordinates", MueLu::NoFactory::getRCP());
  prolongator->SetFactory("Nullspace", MueLu::NoFactory::getRCP());
  prolongator->SetFactory("prolongatorGraph", aggregation);
  prolongator->SetFactory("coarseCoordinatesFineMap", aggregation);
  prolongator->SetFactory("coarseCoordinatesMap", aggregation);
  prolongator->SetFactory("numDimensions", aggregation);
  prolongator->SetFactory("lCoarseNodesPerDim", aggregation);
  prolongator->SetFactory("structuredInterpolationOrder", aggregation);

  coarseLevel.Request("P", prolongator.get());
  coarseLevel.Request(*prolongator);
  prolongator->Build(fineLevel, coarseLevel);

  data.P = coarseLevel.Get<Teuchos::RCP<Matrix> >(
      "P", prolongator.get());
  data.coarseNodesPerDim = fineLevel.Get<Teuchos::Array<LO> >(
      "lCoarseNodesPerDim", aggregation.get());
  return data;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
computeStructuredRAP(
    const Elasticity2DData<Scalar, LocalOrdinal, GlobalOrdinal, Node>& data,
    const int coarseningRate) {
  using SC        = Scalar;
  using LO        = LocalOrdinal;
  using GO        = GlobalOrdinal;
  using NO        = Node;
  using Matrix    = Xpetra::Matrix<SC, LO, GO, NO>;
  using RAPFactory   = MueLu::StructuredRAPFactory<SC, LO, GO, NO>;
  using StencilOffset = typename RAPFactory::StencilOffset;

  RAPFactory rapFactory;
  typename RAPFactory::StructuredGraphSpec graphSpec;
  graphSpec.numDimensions = 2;
  graphSpec.dofsPerNode   = Teuchos::as<LO>(2);
  graphSpec.description   = "Elasticity2D unit-test full stencil";
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      graphSpec.stencilOffsets.push_back(
          StencilOffset{dx, dy, 0});

  Teuchos::RCP<Matrix> Ac;
  rapFactory.GetStructuredGraph(Ac, data.P, data.coarseNodesPerDim,
                                graphSpec);

  Teuchos::Array<int> rate(1, coarseningRate);
  Teuchos::RCP<Teuchos::ParameterList> kernelParams =
      Teuchos::rcp(new Teuchos::ParameterList());
  MueLu::Details::StructuredRAPKernel<SC, LO, GO, NO>::Compute(
      *data.A, *data.P, *Ac, "Elasticity2D", 0, 2,
      data.fineNodesPerDim, data.coarseNodesPerDim, rate, kernelParams);
  return Ac;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >
computeXpetraRAP(
    const Elasticity2DData<Scalar, LocalOrdinal, GlobalOrdinal, Node>& data) {
  using SC     = Scalar;
  using LO     = LocalOrdinal;
  using GO     = GlobalOrdinal;
  using NO     = Node;
  using Matrix = Xpetra::Matrix<SC, LO, GO, NO>;

  Teuchos::RCP<Matrix> Ac = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(
      data.P->getDomainMap(), Teuchos::as<LO>(0));
  Teuchos::RCP<Teuchos::ParameterList> params =
      Teuchos::rcp(new Teuchos::ParameterList());
  Xpetra::TripleMatrixMultiply<SC, LO, GO, NO>::MultiplyRAP(
      *data.P, true, *data.A, false, *data.P, false, *Ac, true, true,
      "MueLu StructuredRAPKernel unit-test reference", params);
  return Ac;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
typename Teuchos::ScalarTraits<Scalar>::magnitudeType
relativeFrobeniusError(
    const Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& actual,
    const Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& expected,
    Teuchos::FancyOStream& out) {
  using SC        = Scalar;
  using LO        = LocalOrdinal;
  using GO        = GlobalOrdinal;
  using NO        = Node;
  using Matrix    = Xpetra::Matrix<SC, LO, GO, NO>;
  using magnitude = typename Teuchos::ScalarTraits<SC>::magnitudeType;

  Teuchos::RCP<Matrix> difference;
  Xpetra::MatrixMatrix<SC, LO, GO, NO>::TwoMatrixAdd(
      *actual, false, Teuchos::ScalarTraits<SC>::one(),
      *expected, false, -Teuchos::ScalarTraits<SC>::one(),
      difference, out);
  if (!difference->isFillComplete())
    difference->fillComplete(expected->getDomainMap(),
                             expected->getRangeMap());

  const magnitude scale = std::max(
      expected->getFrobeniusNorm(), Teuchos::ScalarTraits<magnitude>::one());
  return difference->getFrobeniusNorm() / scale;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void compareConstantElasticity2D(const GlobalOrdinal nx,
                                 const GlobalOrdinal ny,
                                 const int coarseningRate,
                                 Teuchos::FancyOStream& out,
                                 bool& success) {
  using SC        = Scalar;
  using LO        = LocalOrdinal;
  using GO        = GlobalOrdinal;
  using NO        = Node;
  using magnitude = typename Teuchos::ScalarTraits<SC>::magnitudeType;

  const Elasticity2DData<SC, LO, GO, NO> data =
      buildElasticity2DData<SC, LO, GO, NO>(nx, ny, coarseningRate);
  const Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO> > structuredAc =
      computeStructuredRAP<SC, LO, GO, NO>(data, coarseningRate);
  const Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO> > xpetraAc =
      computeXpetraRAP<SC, LO, GO, NO>(data);

  const magnitude error = relativeFrobeniusError<SC, LO, GO, NO>(
      structuredAc, xpetraAc, out);
  const magnitude tolerance =
      magnitude(1000) * Teuchos::ScalarTraits<magnitude>::eps();
  out << "Elasticity2D " << nx << "x" << ny
      << ", coarsening rate " << coarseningRate
      << ": relative Frobenius error = " << error << std::endl;
  TEST_COMPARE(error, <=, tolerance);
}

}  // namespace

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(
    StructuredKernel, ConstantElasticity2DMatchesXpetra,
    Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

  RCP<const Teuchos::Comm<int> > comm =
      TestHelpers::Parameters::getDefaultComm();
  constexpr bool hostMemory =
      std::is_same<typename NO::device_type::memory_space,
                   Kokkos::HostSpace>::value;
  if (comm->getSize() != 1 || !hostMemory ||
      TestHelpers::Parameters::getLib() != Xpetra::UseTpetra) {
    out << "The specialized Elasticity2D kernel is a serial host-memory Tpetra path; "
           "skipping this specialization test."
        << std::endl;
    return;
  }

  compareConstantElasticity2D<SC, LO, GO, NO>(12, 12, 3, out, success);
  compareConstantElasticity2D<SC, LO, GO, NO>(13, 10, 3, out, success);
  compareConstantElasticity2D<SC, LO, GO, NO>(11, 12, 2, out, success);
  compareConstantElasticity2D<SC, LO, GO, NO>(14, 11, 4, out, success);
}

#define MUELU_ETI_GROUP(Scalar, LO, GO, Node)                           \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(                                 \
      StructuredKernel, ConstantElasticity2DMatchesXpetra,              \
      Scalar, LO, GO, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
