# GPGPU

### 1) Requirements:
  - Voor de zekerheid de laatste versie van NVIDIA Driver downloaden: [Download Drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)
  - CUDA Toolkit: [Download CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-downloads)
  
### 2) Run HelloWorld:
  - Open in de map 'HelloWorld' het bestand: 'STL_Reader.sln'
  - Visual Studio: Rebuild Solution in Release mode
  - Ga naar de map `HelloWorld\bin\win64\Release` en open het bestand STL_Reader.exe
  
### 3) Input (Enkel meshes in binair formaat worden ingelezen!):
  1) Het pad van de binnenste mesh wordt gevraagd. Deze zijn te vinden in de map 'Meshes'
  2) Het pad van de buitenste mesh wordt gevraagd.
  3) Het algoritme wordt gevraagd: 0 = ray-triangle uitvoeren, 1 = triangle-triangle uitvoeren
  4) Moet het gekozen algoritme ook op CPU worden uitgevoerd? (JA ==> Ook nog aantal threads zal gevraagd worden (OpenMP))
  
### Map Meshes:

'...L(.stl)' staat voor Large = Dit zijn de vergrote versies.

'...T(.stl)' staat voor Transferred = Dit zijn de verschoven versies.
