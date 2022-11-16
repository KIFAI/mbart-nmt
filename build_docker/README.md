<h1 align="center"><b> Translator Web-App </b></h1>
This is about building docker container image

### 1. Copy Model, WebApp, Server and Translate(Inference) Project
```
$ ./prepare_project.sh
```

### 2. Build docker
```
$ docker build . -t translator:app
$ ./deploy_translator.sh
$ docker exec -it translator /bin/bash
```

### 3. Build cuda-toolkit
```
# You should run install_cuda_toolkit shell script from using "source" command
$ source install_cuda_toolkit.sh 
#Type into "accept"
┌──────────────────────────────────────────────────────────────────────────────┐
│  End User License Agreement                                                  │
│  --------------------------                                                  │
│                                                                              │
│  The CUDA Toolkit End User License Agreement applies to the                  │
│  NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA                    │
│  Display Driver, NVIDIA Nsight tools (Visual Studio Edition),                │
│  and the associated documentation on CUDA APIs, programming                  │
│  model and development tools. If you do not agree with the                   │
│  terms and conditions of the license agreement, then do not                  │
│  download or use the software.                                               │
│                                                                              │
│  Last updated: Nov 2, 2020.                                                  │
│                                                                              │
│                                                                              │
│  Preface                                                                     │
│  -------                                                                     │
│                                                                              │
│  The Software License Agreement in Chapter 1 and the Supplement              │
│  in Chapter 2 contain license terms and conditions that govern               │
│  the use of NVIDIA software. By accepting this agreement, you                │
│──────────────────────────────────────────────────────────────────────────────│
│ Do you accept the above EULA? (accept/decline/quit):                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

#Uncheck "Driver" and Enter "install"
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 460.27.04                                                           │
│ + [X] CUDA Toolkit 11.2                                                      │
│   [X] CUDA Samples 11.2                                                      │
│   [X] CUDA Demo Suite 11.2                                                   │
│   [X] CUDA Documentation 11.2                                                │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘

===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.2/
Samples:  Installed in /root/, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-11.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.2/lib64, or, add /usr/local/cuda-11.2/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.2/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 460.00 is required for CUDA 11.2 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2020 NVIDIA Corporation Built on Mon_Nov_30_19:08:53_PST_2020 Cuda compilation tools, release 11.2, V11.2.67 Build cuda_11.2.r11.2/compiler.29373293_0
```
### 3. Check nvcc
```
$ nvcc --version
```

