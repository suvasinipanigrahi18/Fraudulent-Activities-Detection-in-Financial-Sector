import numpy as np
import time



def TSO(T, fobj, LB, UB, Max_iter):
    Particles_no, Dim = T.shape[0], T.shape[1]
    Tuna1 = np.zeros((1, Dim))
    Tuna1_fit = np.inf
    Low = LB[0, :]
    Up = UB[0, :]
    Iter = 0
    aa = 0.7
    z = 0.05
    Convergence_curve = np.zeros(Max_iter)
    ct = time.time()
    while Iter < Max_iter:
        C = Iter / Max_iter
        a1 = aa + (1 - aa) * C
        a2 = (1 - aa) - (1 - aa) * C

        for i in range(T.shape[0]):
            Flag4ub = T[i, :] > Up
            Flag4lb = T[i, :] < Low
            T[i, :] = (T[i, :] * (~(Flag4ub + Flag4lb))) + (Up * Flag4ub) + (Low * Flag4lb)

            fitness = fobj(T[i, :])

            if fitness < Tuna1_fit:
                Tuna1_fit = fitness
                Tuna1 = T[i, :]

        if Iter == 0:
            fit_old = fitness
            C_old = T.copy()

        for i in range(Particles_no):
            if fit_old[i] < fitness[i]:
                fitness[i] = fit_old[i]
                T[i, :] = C_old[i, :]

        C_old = T.copy()
        fit_old = fitness

        t = (1 - Iter / Max_iter) ** (Iter / Max_iter)

        if np.random.rand() < z:
            T[0, :] = (Up - Low) * np.random.rand() + Low
        else:
            if 0.5 < np.random.rand():
                r1 = np.random.rand()
                Beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((Max_iter - Iter + 1) / Max_iter)))) * (
                    np.cos(2 * np.pi * r1))
                if C > np.random.rand():
                    T[0, :] = a1 * (Tuna1 + Beta * np.abs(Tuna1 - T[0, :])) + a2 * T[0, :]  # Equation (8.3)
                else:
                    IndivRand = np.random.rand(1, Dim) * (Up - Low) + Low
                    T[0, :] = a1 * (IndivRand + Beta * np.abs(IndivRand - T[i, :])) + a2 * T[0, :]  # Equation (8.1)
            else:
                TF = (np.random.rand() > 0.5) * 2 - 1
                if 0.5 > np.random.rand():
                    T[0, :] = Tuna1 + np.random.rand(1, Dim) * (Tuna1 - T[0, :]) + TF * t ** 2 * (
                                Tuna1 - T[0, :])  # Equation (9.1)
                else:
                    T[0, :] = TF * t ** 2 * T[0, :]  # Equation (9.2)

        for i in range(1, Particles_no):
            if np.random.rand() < z:
                T[i, :] = (Up - Low) * np.random.rand() + Low
            else:
                if 0.5 < np.random.rand():
                    r1 = np.random.rand()
                    Beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((Max_iter - Iter + 1) / Max_iter)))) * (
                        np.cos(2 * np.pi * r1))
                    if C > np.random.rand():
                        T[i, :] = a1 * (Tuna1 + Beta * np.abs(Tuna1 - T[i, :])) + a2 * T[i - 1, :]  # Equation (8.4)
                    else:
                        IndivRand = np.random.rand(1, Dim) * (Up - Low) + Low
                        T[i, :] = a1 * (IndivRand + Beta * np.abs(IndivRand - T[i, :])) + a2 * T[i - 1,
                                                                                               :]  # Equation (8.2)
                else:
                    TF = (np.random.rand() > 0.5) * 2 - 1
                    if 0.5 > np.random.rand():
                        T[i, :] = Tuna1 + np.random.rand(1, Dim) * (Tuna1 - T[i, :]) + TF * t ** 2 * (
                                    Tuna1 - T[i, :])  # Equation (9.1)
                    else:
                        T[i, :] = TF * t ** 2 * T[i, :]  # Equation (9.2)

        Iter += 1
        Convergence_curve[Iter] = Tuna1_fit
    ct = time.time() - ct
    return Tuna1_fit, Convergence_curve, Tuna1, ct
