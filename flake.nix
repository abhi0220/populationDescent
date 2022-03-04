{

  description = "A reproducible environment for learning certifiable controllers";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/02336c5c5f719cd6bd4cfc5a091a1ccee6f06b1d";
    mach-nix.url = github:DavHau/mach-nix;
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (inputs.nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
                  

          mach-nix-utils = import inputs.mach-nix {
            inherit pkgs;
            python = "python39Full";
            #pypiDataRev = "e18f4c312ce4bcdd20a7b9e861b3c5eb7cac22c4";
            #pypiDataSha256= "sha256-DmrRc4Y0GbxlORsmIDhj8gtmW1iO8/44bShAyvz6bHk=";
          };

          
          python-with-deps = mach-nix-utils.mkPython {

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
