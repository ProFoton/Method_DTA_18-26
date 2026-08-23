I. МАТЕМАТИЧЕСКАЯ МОДЕЛЬ ЗАТУХАНИЯ (DECAY FUNCTION)
Протокол управляет тремя переменными вывода на основе кумулятивного индекса когнитивной нагрузки (CLI — Cognitive Load Index):\(I_{delay}\) — искусственная межтокенная задержка (инжекция миллисекунд в поток генерации SSE).\(V_{sat}\) — коэффициент насыщенности интерфейса (цветовая гамма UI).\(C_{max}\) — лимит контекстного окна или жесткий порог прерывания генерации.1. Расчет Когнитивной Нагрузки (CLI)Каждую секунду сессии CLI пересчитывается по формуле:\(CLI_{t}=CLI_{t-1}+\left(\omega _{1}\cdot \frac{\Delta T_{session}}{T_{max}}+\omega _{2}\cdot BR_{stress}\right)-\omega _{3}\cdot \Delta T_{idle}\)Где:\(\Delta T_{session}\): непрерывное время сессии в секундах.\(T_{max}\): критический порог непрерывного внимания (по умолчанию — 1800 секунд / 30 минут).\(BR_{stress}\): нормированный показатель биометрического стресса от rPPG (от 0.0 до 1.0).\(\Delta T_{idle}\): время отсутствия активности (паузы).ω₁, ω₂, ω₃: весовые коэффициенты настройки чувствительности.2. Динамическая задержка токенов (Token Pacing)Задержка перед отправкой каждого следующего токена \(t_{delay}\) рассчитывается через логистическую функцию, предотвращая резкий «когнитивный удар» (резкую блокировку), который вызывает фрустрацию:\(t_{delay}(CLI)=t_{min}+\frac{t_{max}-t_{min}}{1+e^{-k\cdot (CLI-CLI_{thresh})}}\)При CLI → 0: задержка минимальна (\(t_{min} \approx 20\text{мс}\)).При CLI → 1: задержка экспоненциально растет до \(t_{max} \approx 800\text{мс}\), симулируя «засыпание» интерфейса и делая чтение некомфортным.II. КОНЕЧНЫЙ АВТОМАТ СОСТОЯНИЙ (STATE MACHINE)       [ CLI < 0.5 ]                [ CLI >= 0.5 ]               [ CLI >= 0.85 ]
+-------------------------+   +-------------------------+   +-------------------------+

|     STATE: ACTIVE       |   |    STATE: COMPRESSION   |   |     STATE: ECLIPSE      |
|-------------------------|   |-------------------------|   |-------------------------|
| Delay: 15-30ms          |-->| Delay: 100-300ms        |-->| Delay: 500-1200ms       |
| UI: Full RGB            |   | UI: Grayscale 50%       |   | UI: Monochrom / Dark    |
| Content: High Density   |   | Content: Summarization  |   | Content: Hard Stop Prompt|
+-------------------------+   +-------------------------+   +-------------------------+

                                           |                             |
                                   [ User Leaves ]               [ Hard Session Lock ]
                                           v                             v
                              +-------------------------------------------------------+

                              |                  STATE: REALITY_RESET                 |
                              |-------------------------------------------------------|
                              | Cool-down Lock: 900s                                  |
                              | Action: Termination of Web Socket / Inference Process |
                              +-------------------------------------------------------+
III. АРХИТЕКТУРНЫЙ ПСЕВДОКОД РЕАЛИЗАЦИИ1. Модуль Инференса (Backend / LLM Wrapper)Этот компонент перехватывает поток генерации токенов (Stream) и искусственно модулирует задержки на уровне шины данных.pythonimport time
import math

class SunrisePacingController:
    def __init__(self, t_min=0.02, t_max=0.8, k=10, thresh=0.7):
        self.t_min = t_min      # Минимальная задержка (20мс)
        self.t_max = t_max      # Максимальная задержка (800мс)
        self.k = k              # Крутизна кривой задержки
        self.thresh = thresh    # Порог триггера
        
    def calculate_token_delay(self, current_cli: float) -> float:
        """Расчет задержки перед отправкой следующего токена"""
        if current_cli <= 0.3:
            return self.t_min
        
        # Логистическая модуляция для плавного замедления
        delay = self.t_min + (self.t_max - self.t_min) / (1 + math.exp(-self.k * (current_cli - self.thresh)))
        return delay

    def execute_stream_pacing(self, token_generator, user_metrics_provider):
        """Перехват генерации и отгрузка в канал с задержкой"""
        for token in token_generator:
            # Считывание актуального CLI из общего стейта пользователя
            cli = user_metrics_provider.get_current_cli()
            
            # Если достигнут терминальный порог — жесткое закрытие сессии
            if cli >= 0.95:
                yield "[SYSTEM: Сессия переведена в режим полной разгрузки. Экран заблокирован. Вернитесь в физическое пространство.]"
                break
                
            delay = self.calculate_token_delay(cli)
            time.sleep(delay) # Контролируемый простой потока
            yield token
Используйте код с осторожностью.2. Модуль Управления Интерфейсом (Frontend Renderer)CSS-инжекция и манипуляция DOM для снижения дофаминового раздражения зрительного нерва.javascriptclass SunriseUIController {
    constructor() {
        this.rootHtml = document.documentElement;
        this.sessionStartTime = Date.now();
    }

    // Вызывается каждые 500мс через Worker
    updateUIConfiguration(cli) {
        // 1. Десатурация (Обесцвечивание интерфейса)
        // При CLI = 0.5 -> filter(grayscale(0%)), При CLI = 0.85 -> filter(grayscale(100%))
        let grayscaleValue = 0;
        if (cli > 0.5) {
            grayscaleValue = Math.min(((cli - 0.5) / 0.35) * 100, 100);
        }
        this.rootHtml.style.setProperty('--dta-grayscale', `${grayscaleValue}%`);

        // 2. Снижение контраста и яркости (Эмуляция сумерек)
        let contrastValue = 100;
        if (cli > 0.6) {
            contrastValue = 100 - Math.min(((cli - 0.6) / 0.3) * 30, 30); // Падение до 70%
        }
        this.rootHtml.style.setProperty('--dta-contrast', `${contrastValue}%`);

        // 3. Динамический CSS-инжект в реальном времени
        this.rootHtml.style.filter = `grayscale(${grayscaleValue}%) contrast(${contrastValue}%)`;
        
        // 4. Подавление анимаций (Убираем микровсплытия, скролл-эффекты)
        if (cli > 0.7) {
            this.rootHtml.classList.add('dta-disable-animations');
        }
    }
}
