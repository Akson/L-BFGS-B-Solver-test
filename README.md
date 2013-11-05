L-BFGS-B-Solver-test
====================

Test for CUDA version of the L-BFGS-B solver.

This project was created for testing the L-BFGS-B solver on high-end NVidia card due to a problem with running of it on the NVidia Titan card.

Requirements (it was tested on similar config):
1) CUDA 5.5
2) MSVS 2012
3) NVidia GPU (main target is the GK110 like Titan or GTX780).
4) Windows 7
5) NVidia drivers 320.57

How to build and run:
1) Open ConsoleApplication2.sln
2) Build solution
3) Run
4) If it does not work, check CUDA paths. Some of them are hardcoded in project settings.

How to run prebuilt executable:
1) Go to Data folder and ConsoleApplication2.exe


Expected results:
Starting LBFGS-B Test...
iteration: 0
iteration: 1
iteration: 2
iteration: 3
iteration: 4
iteration: 5
iteration: 6
iteration: 7
iteration: 8
iteration: 9
iteration: 10
iteration: 11
iteration: 12
Test PASSED: max weight error = 0.03243 at index 0
Press any key to continue . . .
