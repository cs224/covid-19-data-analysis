

* use for Germany the RKI data in general
* write code in covid19.py to take for selected countries the official data and not the JHU data.
* extend fitSigExt to several hammer-dance scenarios (using constraints in the optimizer)
  * estimate R_0 based on the fitSig part of fitSigExt
* estimate R_0 using kalman filtering
* fit asymmetric upward and downward slopes
* use gaussian-process regression on total number to predict near term future (using gpflow and week periodicity)
