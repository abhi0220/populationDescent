{

  description = "A reproducible environment for learning certifiable controllers";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-21.11";
    mach-nix.url = github:DavHau/mach-nix;
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (inputs.nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
          
          python-with-deps = inputs.mach-nix.lib."${system}".mkPython {
            providers = {
              tensorflwo="nixpkgs";
            };

            requirements=''
              numpy
              matplotlib
              future
              tensorflow
              scipy
              tqdm
              mypy
              noise
              pylint
              tensorflow-probability
              tensorflow-addons
            '';
          };

      in {
        devShell = pkgs.mkShell {
          buildInputs=[
            python-with-deps
          ];
        };
      }
    );
}
