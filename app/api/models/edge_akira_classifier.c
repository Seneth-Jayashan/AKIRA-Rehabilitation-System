/*
 * AKIRA Edge-AI Inference Engine (Generated via m2cgen)
 * Real-time Dual-IMU Kinematic Classifier for Knee Rehabilitation
 * Target: Embedded Microcontrollers (ESP32 / ARM Cortex-M4 / nRF5340)
 * Inputs: double input[100] (84 sequence features + 16 kinematic features)
 * Output 1: Exercise Class (0=Background, 1=Seated Knee Extension, 2=Mini Squat)
 * Output 2: Quality Grade (0=Good, 1=Needs Improvement, 2=Bad)
 */
#include <math.h>
#include <stdint.h>

#include <string.h>
void akira_classify_exercise(double * input, double * output) {
    double var0[3];
    if (input[38] <= 0.5377931892871857) {
        if (input[21] <= 35.04867744445801) {
            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
        } else {
            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
        }
    } else {
        if (input[98] <= 65.60392761230469) {
            if (input[28] <= 65.8023681640625) {
                if (input[89] <= -26.210350036621094) {
                    memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                } else {
                    if (input[24] <= 0.5446777045726776) {
                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                    } else {
                        if (input[61] <= 0.06372998654842377) {
                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                        }
                    }
                }
            } else {
                memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
            }
        } else {
            if (input[89] <= -7.982404947280884) {
                memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
            } else {
                if (input[87] <= 229.81080627441406) {
                    if (input[76] <= 0.01256281416863203) {
                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                    }
                } else {
                    memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                }
            }
        }
    }
    memcpy(output, var0, 3 * sizeof(double));
}


#include <string.h>
void akira_assess_quality(double * input, double * output) {
    double var0[3];
    if (input[97] <= 26.76460361480713) {
        if (input[12] <= 0.069785475730896) {
            if (input[31] <= -0.08752340450882912) {
                if (input[18] <= 0.0640723928809166) {
                    if (input[45] <= -2.0930907130241394) {
                        if (input[6] <= -0.1417551003396511) {
                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                        }
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                    }
                } else {
                    if (input[25] <= 0.1900954246520996) {
                        if (input[14] <= 0.012268454767763615) {
                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.037037037037037035, 0.9259259259259259, 0.037037037037037035}, 3 * sizeof(double));
                        }
                    } else {
                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                    }
                }
            } else {
                if (input[0] <= 0.11013734340667725) {
                    memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                } else {
                    memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                }
            }
        } else {
            if (input[59] <= 244.96351623535156) {
                if (input[94] <= 26.70300006866455) {
                    memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                } else {
                    memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                }
            } else {
                memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
            }
        }
    } else {
        if (input[57] <= 109.45669555664062) {
            if (input[88] <= 52.53104591369629) {
                if (input[14] <= 0.4271628111600876) {
                    if (input[40] <= -49.377349853515625) {
                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                    } else {
                        if (input[91] <= 2.2324999570846558) {
                            memcpy(var0, (double[]){0.5121951219512195, 0.024390243902439025, 0.4634146341463415}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.11267605633802817, 0.028169014084507043, 0.8591549295774648}, 3 * sizeof(double));
                        }
                    }
                } else {
                    if (input[53] <= 44.02553939819336) {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                    } else {
                        if (input[54] <= 0.48517249524593353) {
                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                        }
                    }
                }
            } else {
                if (input[89] <= -10.579365730285645) {
                    if (input[96] <= -176.31352996826172) {
                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0}, 3 * sizeof(double));
                    }
                } else {
                    if (input[5] <= -6.195142030715942) {
                        memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                    } else {
                        if (input[93] <= 148.47154998779297) {
                            memcpy(var0, (double[]){0.7476635514018691, 0.0, 0.2523364485981308}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                        }
                    }
                }
            }
        } else {
            if (input[68] <= 0.5658102929592133) {
                if (input[96] <= -67.32229232788086) {
                    if (input[15] <= 5.275750398635864) {
                        memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                    } else {
                        if (input[46] <= -8.718945503234863) {
                            memcpy(var0, (double[]){0.9024390243902439, 0.04878048780487805, 0.04878048780487805}, 3 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){0.0, 1.0, 0.0}, 3 * sizeof(double));
                        }
                    }
                } else {
                    memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
                }
            } else {
                memcpy(var0, (double[]){0.0, 0.0, 1.0}, 3 * sizeof(double));
            }
        }
    }
    memcpy(output, var0, 3 * sizeof(double));
}
