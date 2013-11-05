L-BFGS-B-Solver-test
====================

Test for CUDA version of the L-BFGS-B solver.<br>
<br>
This project was created for testing the L-BFGS-B solver on high-end NVidia card due to a problem with running of it on the NVidia Titan card.<br>
<br>
Requirements (it was tested on similar config):<br>
1) CUDA 5.5<br>
2) MSVS 2012<br>
3) NVidia GPU (main target is the GK110 like Titan or GTX780).<br>
4) Windows 7<br>
5) NVidia drivers 320.57<br>
<br>
How to build and run:<br>
1) Open ConsoleApplication2.sln<br>
2) Build solution<br>
3) Run<br>
4) If it does not work, check CUDA paths. Some of them are hardcoded in project settings.<br>
<br>
How to run prebuilt executable:<br>
1) Go to Data folder and ConsoleApplication2.exe<br>
<br>
<br>
Expected results:<br>
Starting LBFGS-B Test...<br>
iteration: 0<br>
iteration: 1<br>
iteration: 2<br>
iteration: 3<br>
iteration: 4<br>
iteration: 5<br>
iteration: 6<br>
iteration: 7<br>
iteration: 8<br>
iteration: 9<br>
iteration: 10<br>
iteration: 11<br>
iteration: 12<br>
Test PASSED: max weight error = 0.03243 at index 0<br>
Press any key to continue . . .<br>
