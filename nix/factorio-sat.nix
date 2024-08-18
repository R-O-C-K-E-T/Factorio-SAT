{ sourceRoot ? ../., lib, buildPythonApplication, makeWrapper, setuptools
, pygame, pillow, pyopengl, graphviz, ffmpeg-python, numpy, python-sat
, luaparser, }:
let
  pyproject = lib.importTOML "${sourceRoot}/pyproject.toml";
  version = pyproject.project.version;

  self = buildPythonApplication {
    pname = pyproject.project.name;
    inherit version;

    src = sourceRoot;
    pyproject = true;

    nativeBuildInputs = [ setuptools makeWrapper ];

    dependencies = [
      numpy
      python-sat
      luaparser
      (pygame.overrideAttrs {
        # evidently broken at the moment
        installCheckPhase = "";
      })
      pillow
      pyopengl
      graphviz
      ffmpeg-python
    ];

    preBuild = ''
      sed -i 's/python-sat/python-sat==${python-sat.version}/g' pyproject.toml
    '';

    meta = {
      inherit (pyproject.project.urls) homepage;
      inherit (pyproject.project) description;
      license = lib.licenses.gpl3Plus;
      platforms = lib.platforms.all;
      # mainProgram = there is none, use the factorio-sat-cli wrapper
      maintainers = [ lib.maintainers.spikespaz ];
    };
  };
in self
