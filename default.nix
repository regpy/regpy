{
  nixpkgs ? import nix/fetch.nix {
    name = "nixpkgs";
    url = https://github.com/NixOS/nixpkgs/archive/1601f559e89ba71091faa26888711d4dd24c2d4d.tar.gz;
    sha256 = "00rwjxjx42nbiqc1qyp8lpm4vfq3lgxc6ikfd2fjp9bnj15dgd35";
  }
, pkgs ? import nixpkgs {}
, ngsolve ? false
} @ args:

with pkgs;
with lib;

let

  runscript = env: writeScript "run" ''
    #! ${stdenvNoCC.shell}
    PATH=${env}/bin:$PATH
    (($#)) || set -- "''${SHELL:-${bashInteractive}/bin/bash}"
    exec "$@"
  '';

in rec {
  inherit nixpkgs;

  ngsolve = callPackage nix/ngsolve.nix {};

  pdoc3 = python3.pkgs.callPackage nix/pdoc3.nix {
    mako = python3.pkgs.callPackage nix/mako.nix {};
    markdown3 = python3.pkgs.callPackage nix/markdown3.nix {};
  };

  env = python3.buildEnv.override {
    extraLibs = (with python3.pkgs; [
      numpy scipy matplotlib
    ])
    ++ (optional args.ngsolve ngsolve);
  };

  devenv = env.override (attrs: {
    extraLibs = attrs.extraLibs ++ (with python3.pkgs; [
      ipython pdoc3 pytest
    ]);
  });

  run = runscript env;

  rundev = runscript devenv;
}
