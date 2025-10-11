import numpy as np

class Golden:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        # النسبة الذهبية φ
        self.phi = (1 + np.sqrt(5)) / 2
        
        # معاملات الزخم والتكيّف مبنية على النسبة الذهبية
        self.beta1 = 1 - 1 / (self.phi ** 2)   # ≈ 0.618
        self.beta2 = 1 - 1 / (self.phi ** 3)   # ≈ 0.764
        
        # معدل التعلم الابتدائي
        self.lr = learning_rate
        self.epsilon = epsilon
        
        # المتغيرات الداخلية
        self.m = None  # المتوسط الأول (الزخم)
        self.v = None  # المتوسط الثاني (التباين)
        self.t = 0     # عدد الخطوات (iteration counter)

    def update(self, params, grads, total_steps=1000):
        """
        params: الأوزان الحالية (numpy array)
        grads: التدرجات (numpy array)
        total_steps: العدد الكلي للخطوات لتعديل معدل التعلم الذهبي
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # زيادة عدد الخطوات
        self.t += 1

        # تحديث المتوسطات
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # تصحيح التحيز
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # معدل التعلم الذهبي الديناميكي
        golden_lr = self.lr * (1 / (self.phi ** (self.t / total_steps)))

        # تحديث الأوزان
        params = params - golden_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params
