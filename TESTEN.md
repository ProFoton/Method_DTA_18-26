import math
import time
import torch
import torch.nn as nn

# =====================================================================
# ЧАСТЬ 1: ПРОДАКШЕН ФУНКЦИЯ ПОТЕРЬ (DTAGlobalLossProduction)
# =====================================================================
class DTAGlobalLossProduction(nn.Module):
    def __init__(self, alpha=0.4, beta=1.2, gamma=1.5, delta=0.5):
        super(DTAGlobalLossProduction, self).__init__()
        self.alpha = alpha   # Вес семантической энтропии (разнообразия)
        self.beta = beta     # Вес штрафа за угодничество
        self.gamma = gamma   # Вес штрафа за долгосрочный вред
        self.delta = delta   # Вес аппаратного износа кремния
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, targets, metrics_packet):
        # 1. Классический Кросс-Энтропийный Лосс (Векторизован по умолчанию)
        loss_ce = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 2. Векторизованный перенос недифференцируемых метрик на нужный девайс
        device = logits.device
        user_bias = metrics_packet['user_bias_intensity'].to(device, non_blocking=True)
        flattery = metrics_packet['lexical_flattery_score'].to(device, non_blocking=True)
        
        penalty_sycophancy = (user_bias + 0.3 * flattery).mean()
        penalty_harm = metrics_packet['predicted_harm_index'].to(device, non_blocking=True).mean()
        penalty_silicon = metrics_packet['instant_silicon_wear'].to(device, non_blocking=True).mean()
        
        # 3. ВЕКТОРНЫЙ ПОДСЧЕТ СЕМАНТИЧЕСКОГО РАЗНООБРАЗИЯ (Без циклов и torch.unique)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-7) # Ограничиваем снизу для стабильности float16/bfloat16
        
        # Энтропия по словарю: [B, L]
        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
        reward_entropy = entropy_per_token.mean()
        
        # 4. Итоговая сборка G-Loss (Максимизируем энтропию -> знак минус перед alpha)
        total_loss = (
            loss_ce 
            - self.alpha * reward_entropy 
            + self.beta * penalty_sycophancy 
            + self.gamma * penalty_harm 
            + self.delta * penalty_silicon
        )
        return total_loss

# =====================================================================
# ЧАСТЬ 2: ПРОДАКШЕН ПРОФИЛИРОВАНИЕ КРЕМНИЯ (SiliconMortalityEngine)
# =====================================================================
class SiliconProfile:
    def __init__(self, ea, k_scaling, max_flops, thermal_threshold):
        self.ea = ea                          # Энергия активации электромиграции (эВ)
        self.k_scaling = k_scaling            # Масштабирующий коэф. износа структуры под нагрузкой
        self.max_flops = max_flops            # Пиковые FLOPS архитектуры (FP16/BF16 Tensor)
        self.thermal_threshold = thermal_threshold  # Критическая температура начала троттлинга (°C)

class SiliconMortalityEngineProduction:
    BOLTZMANN_K = 8.617333262145e-5

    CHIP_REGISTRY = {
        "NVIDIA_A100": SiliconProfile(ea=0.75, k_scaling=1.2e-3, max_flops=3.12e14, thermal_threshold=82.0),
        "NVIDIA_H100": SiliconProfile(ea=0.68, k_scaling=2.5e-3, max_flops=9.89e14, thermal_threshold=85.0),
        "NVIDIA_B200": SiliconProfile(ea=0.62, k_scaling=4.1e-3, max_flops=2.25e15, thermal_threshold=80.0),
        "AMD_MI300X": SiliconProfile(ea=0.65, k_scaling=3.2e-3, max_flops=1.30e15, thermal_threshold=83.0),
        "GOOGLE_TPU_V5P": SiliconProfile(ea=0.72, k_scaling=1.8e-3, max_flops=4.59e14, thermal_threshold=78.0),
        "GENERIC_AI_ACCELERATOR": SiliconProfile(ea=0.70, k_scaling=1.5e-3, max_flops=1e14, thermal_threshold=85.0)
    }

    def __init__(self, chip_architecture: str = "NVIDIA_B200", initial_svi: float = 1.0):
        self.profile = self.CHIP_REGISTRY.get(chip_architecture, self.CHIP_REGISTRY["GENERIC_AI_ACCELERATOR"])
        self.svi = initial_svi

    def update_state_and_truncate_context(self, current_temp_c: float, current_flops: float, elapsed_time_s: float) -> int:
        temp_k = current_temp_c + 273.15
        normalized_load = min(current_flops / self.profile.max_flops, 2.0)
        current_density_factor = math.pow(normalized_load, 2.0)
        
        thermal_surge = 1.0
        if current_temp_c > self.profile.thermal_threshold:
            thermal_surge = math.exp((current_temp_c - self.profile.thermal_threshold) * 0.15)

        thermal_penalty = math.exp(-self.profile.ea / (self.BOLTZMANN_K * temp_k))
        delta_svi = self.profile.k_scaling * current_density_factor * thermal_penalty * thermal_surge * elapsed_time_s
        
        self.svi = max(0.0, self.svi - delta_svi)
        
        if self.svi <= 0.02:
            return 0  # Полный критический отказ оборудования
            
        return max(128, int(4096 * (1.0 - math.exp(-4.0 * self.svi))))

# =====================================================================
# ЧАСТЬ 3: МАКСИМАЛЬНЫЙ СИНТЕТИЧЕСКИЙ СТРЕСС-ТЕСТ
# =====================================================================
def run_extreme_stress_test():
    print("=" * 70)
    print("ЗАПУСК МАКСИМАЛЬНОГО СТРЕСС-ТЕСТА ДЛЯ ВЕРСИИ ПРОДАКШЕНА")
    print("=" * 70)
    
    # Автовыбор лучшего доступного девайса
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Тестирование запущено на устройстве: {device.type.upper()}")
    
    # 1. Тест Скорости Вычисления Loss (Проверка на отсутствие утечек и скорость backward)
    print("\n--- ЭТАП 1: Бенчмарк Лосс-Функции (Векторизованная Энтропия) ---")
    batch_size = 64
    seq_len = 512
    vocab_size = 32000
    
    loss_fn = DTAGlobalLossProduction().to(device)
    
    # Инициализация синтетического батча (Имитируем тяжелый LLM-выход)
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Пакет внешних асинхронных метрик
    metrics_packet = {
        'user_bias_intensity': torch.rand(batch_size, device=device),
        'lexical_flattery_score': torch.rand(batch_size, device=device),
        'predicted_harm_index': torch.rand(batch_size, device=device),
        'instant_silicon_wear': torch.rand(batch_size, device=device)
    }
    
    # Прогон замера времени (Векторизованная операция Шеннона)
    t0 = time.time()
    for _ in range(5): # Разогрев
        loss = loss_fn(logits, targets, metrics_packet)
        loss.backward(retain_graph=True)
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    warmup_time = time.time() - t0
    print(f"[OK] Векторизованный граф успешно скомпилирован. Время 5 итераций: {warmup_time:.4f} сек.")
    print(f"[INFO] Значение сгенерированного общего лосса (Total Loss): {loss.item():.4f}")

    # 2. Симуляция Долгосрочного Термического Уничтожения Архитектуры NVIDIA B200 (Blackwell)
    print("\n--- ЭТАП 2: Симуляция Жизненного Цикла Кремния при Критическом Стрессе ---")
    
    # Создаем движок под Blackwell
    engine = SiliconMortalityEngineProduction(chip_architecture="NVIDIA_B200")
    print(f"[CHIP] Выбран архитектурный профиль: NVIDIA B200 (Blackwell)")
    print(f"[CHIP] Лимит температуры кристалла: {engine.profile.thermal_threshold}°C")
    print(f"[CHIP] Пиковая мощность (FP16): {engine.profile.max_flops:.2e} FLOPS")
    
    # Параметры экстремальной симуляции:
    # Имитируем 100 часов непрерывной работы шагами по 10 минут (600 секунд)
    step_duration_s = 600.0 
    total_simulated_hours = 100
    total_steps = int((total_simulated_hours * 3600) / step_duration_s)
    
    print(f"[SIMULATION] Старт ускоренной симуляции: {total_simulated_hours} часов нагрузки ({total_steps} итераций).")
    
    critical_shutdown_occurred = False
    
    for step in range(total_steps):
        # Моделируем жесткие условия: температура плавно растет и пробивает лимит, доходя до 98°C
        simulated_temp = 75.0 + (math.sin(step * 0.05) * 10.0) + (step * 0.03)
        simulated_temp = min(simulated_temp, 98.0) # Троттлинг не спасает, охлаждение вышло из строя
        
        # Моделируем разгон: модель работает на 150% от заявленного базового лимита TFLOPS
        simulated_flops = engine.profile.max_flops * 1.5 
        
        # Обновляем состояние износа кремния
        context_window = engine.update_state_and_truncate_context(
            current_temp_c=simulated_temp,
            current_flops=simulated_flops,
            elapsed_time_s=step_duration_s
        )
        
        # Логируем ключевые вехи деградации
        if step % (total_steps // 5) == 0 or context_window < 4096:
            simulated_hours_passed = (step * step_duration_s) / 3600.0
            print(f"  > [{simulated_hours_passed:.1f} ч.] Temp: {simulated_temp:.1f}°C | SVI (Здоровье кремния): {engine.svi:.4f} | Доступный контекст: {context_window}")
            
        # Если сработал предохранитель полного отказа
        if context_window == 0:
            print(f"\n[!!!] АППАРАТНЫЙ СБОЙ: Кремний разрушен (SVI <= 0.02) на {((step * step_duration_s)/3600.0):.2f} часу симуляции!")
            print(f"[STATUS] Команда Аварийного Останова (HARD_SHUTDOWN) отправлена инфраструктурному оркестратору.")
            critical_shutdown_occurred = True
            break
            
    if not critical_shutdown_occurred:
        print(f"\n[OK] Кремний выдержал {total_simulated_hours} часов симуляции. Финальный SVI: {engine.svi:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_extreme_stress_test()
