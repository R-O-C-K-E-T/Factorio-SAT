{ lib, stdenv, callPackage, python3, libuuid }:
let
  # This import is a builder extracted from Nixpkgs.
  mkAntlr = callPackage ./mk-antlr.nix { };

  versions."4.7.2" = {

    antlr = (mkAntlr {
      version = "4.7.2";
      sourceSha256 = "sha256-kta+K/c6cUdOuW6jWMpX4tA22GXsaDBwtalzw4z+gN4=";
      jarSha256 = "sha256-aFI4bXl17/KRcdrgAswiMlFRDTXyka4neUjzgaezgLQ=";
      extraCppBuildInputs = lib.optional stdenv.isLinux libuuid;
      extraCppCmakeFlags = [ "-DANTLR4_INSTALL=ON" ];
    }).antlr;

    pythonOverrides = pypkgs: pypkgs0: {
      antlr4-python3-runtime = pypkgs.toPythonModule
        ((pypkgs0.antlr4-python3-runtime.override {
          antlr4 = versions."4.7.2".antlr;
        }).overridePythonAttrs {
          postPatch = "";
          doCheck = false;
        });
    };

  };
in versions
