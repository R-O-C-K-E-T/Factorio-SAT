{ lib, buildPythonPackage, fetchPypi, setuptools-scm, setuptools, six
, multimethod, antlr4, antlr4-python3-runtime }:
buildPythonPackage rec {
  pname = "luaparser";
  version = "3.2.1";
  pyproject = true;
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-7oF6Wc5C/fqRmXoludubGkGRruVIwjeeNqNI4ukCUmk=";
  };
  # I don't know how python dependencies work but this does
  build-system = [ setuptools-scm ];
  dependencies = [ setuptools six antlr4 antlr4-python3-runtime multimethod ];
}
