# Package Managers


To install software packages on different platforms, you need to use the package manager specific to that platform. Here's an explanation for each platform and its package manager:

### **Summary of Package Managers**
| Platform                            | Package Manager                      | Comments                |
|-------------------------------------|--------------------------------------|-------------------------|
| [Debian](debian) :factory:          | APT (APT - Advanced Package Tool)    | Installing required packages and installing and compiling bladeRF host libraries and binaries  |
| [Fedora/CentOS/RHEL](dnf) :factory: | DNF (DNF - Dandified Yum)            | Installing required packages and installing and compiling bladeRF host libraries and binaries  |
| [Ubuntu](ubuntu)    :package:       | APT (APT - Advanced Package Tool)    | Installing required packages provided by Nuand |

These package managers simplify software installation and ensure that dependencies are resolved automatically. Always keep your package list updated (`apt update` or `dnf update`) to get the latest versions and security patches.