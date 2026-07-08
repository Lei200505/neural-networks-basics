import math

def adam_2d(f, grad_f, x1, x2, m1=0.0, m2=0.0, v1=0.0, v2=0.0, t=0,
             eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=1000):

    for _ in range(steps):
        t += 1
        g1, g2 = grad_f(x1, x2)

        # Update biased first moment
        m1 = beta1 * m1 + (1- beta1)* g1
        m2 = beta1 * m2 + (1- beta1)* g2

        # Update biased second moment
        v1 = beta2 * v1 + (1- beta2)* g1**2
        v2 = beta2 * v2 + (1- beta2)* g2**2

        # Bias correction
        m1_hat = m1 / (1 - beta1**t)
        m2_hat = m2 / (1 - beta1**t)
        v1_hat = v1 / (1 - beta2**t)
        v2_hat = v2 / (1 - beta2**t)

        # Parameter update
        x1 -= eta * m1_hat / (math.sqrt(v1_hat)+ eps)
        x2 -= eta * m2_hat / (math.sqrt(v2_hat)+ eps)

    return x1, x2