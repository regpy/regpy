{ buildPythonPackage, fetchPypi, setuptools-git, setuptools_scm, markupsafe
, mako, markdown3 }:

buildPythonPackage rec {
  pname = "pdoc3";
  version = "0.6.3";
  src = fetchPypi {
    inherit pname version;
    sha256 = "18mczzsch2143b7wy05fxf3sbss8rn2jgqxvvfcfk47pn4ggci5i";
  };
  buildInputs = [ setuptools-git setuptools_scm ];
  propagatedBuildInputs = [ mako markdown3 markupsafe ];
}
