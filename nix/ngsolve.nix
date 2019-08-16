{ lib, fetchFromGitHub, python3Packages,
  cmake, tk, tix, openblasCompat, liblapackWithoutAtlas,
  mesa_glu, suitesparse, xorg, makeWrapper,
  writeText, glibcLocales, scalapack
}:

with lib;

python3Packages.buildPythonPackage rec {
  name = "ngsolve-${version}";
  namePrefix = "";
  version = "6.2.1904";
  format = "other";

  src = fetchFromGitHub {
    owner = "NGSolve";
    repo = "ngsolve";
    rev = "v${version}";
    sha256 = "02pr9j3r2xnjvgwwnpc2i0gds2z582h4pshny37nads4czrkzxn2";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake makeWrapper
  ];

  buildInputs = [
    tk tix openblasCompat liblapackWithoutAtlas suitesparse
    mesa_glu xorg.libXmu xorg.libX11 scalapack
  ];

  propagatedBuildInputs = with python3Packages; [
    tkinter
  ];

  patches = [ ./ngsolve.patch ];

  cmakeFlags = [
    "-DUSE_UMFPACK=ON"
    "-DBUILD_UMFPACK=OFF"
    "-DSuiteSparse=${suitesparse}"
    "-DSCALAPACK_LIBRARY=${scalapack}/lib/libscalapack.so"
    "-Dgit_version_string=v${version}-0-g${substring 0 8 src.rev}"
  ];

  postInstall = ''
    chmod +x $out/lib/*.so
    wrapProgram $out/bin/netgen \
      --prefix PYTHONPATH : ".:$out/${python3Packages.python.sitePackages}:$out/lib" \
      --prefix TCLLIBPATH " " "${tix}/lib" \
      --set NETGENDIR "$out/bin" \
      --set LOCALE_ARCHIVE "${glibcLocales}/lib/locale/locale-archive"
  '';

  shellHook = ''
    export PYTHONPATH="$out/lib''${PYTHONPATH:+:}$PYTHONPATH"
    export TCLLIBPATH="${tix}/lib''${TCLLIBPATH:+ }$TCLLIBPATH"
    export NETGENDIR="$out/bin"
    export LOCALE_ARCHIVE="${glibcLocales}/lib/locale/locale-archive"
  '';

}
