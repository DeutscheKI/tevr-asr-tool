lddtree cmake-build-minsizerel/tevr_asr_tool

(cd cmake-build-minsizerel && cpack -G DEB)

dpkg -c cmake-build-minsizerel/tevr_asr_tool-0.1.1-Linux.deb
dpkg -I cmake-build-minsizerel/tevr_asr_tool-0.1.1-Linux.deb
