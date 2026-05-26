# sm56_defoamer_bench
PCIe GEN 1@ 4x transfer speedup hacks compared to Сырой трансферъ FP16

<img width="1141" height="568" alt="image" src="https://github.com/user-attachments/assets/d31a1404-6a5a-43c8-b285-403c5447bf04" />

```
julia sm56_defoamer_bench_v2.0.jl                     
[ Info: 🖥️ GPU: NVIDIA CMP 50HX (SM 7.5, 56 SM)
[ Info: 🧪 SM56-Defoamer Benchmark v2.0
[ Info: 📏 Минимальный размѣръ тензора: 1.0 МБ
[ Info: 🎲 Генерація синтетическихъ активаций (увеличенный размѣръ)
[ Info:   Размѣръ одного тензора: 24.0 МБ
[ Info:   ✅ Сгенерировано 22 тензоровъ, общій объемъ: 528.0 МБ
[ Info: 🔬 Діагностика перваго тензора...
[ Info:   Типъ: CuArray{Float16, 3, CUDA.DeviceMemory}
[ Info:   Размѣръ: (384, 512, 64)
[ Info:   Минимумъ: 0.0
[ Info:   Максимумъ: 0.999
[ Info:   Среднее: 0.4995893
[ Info:   Есть NaN: false
[ Info:   Есть Inf: false
[ Info: 📊 Полезная нагрузка: 528.00 МБ (22 тензоровъ)
[ Info: 🔥 Прогрѣвъ...
[ Info: 📏 Сырой трансферъ FP16
[ Info:    Время: 2.1028 сек | Пропускная способность: 502.2 МБ/сек
[ Info: 📦 Сжатіе SM56-Defoamer INT8
[ Info:    Время: 1.7379 сек | Пропускная способность: 607.6 МБ/сек | Сжатіе: 2.00x | Ускореніе: 1.2x
[ Info: 🔍 Анализъ погрѣшностей (Float32 метрики, защита отъ переполненія)
[ Info: 🐛 Діагностика метрикъ (первый тензоръ):
[ Info:   sum_sq_sig: 4.189633e6
[ Info:   sum_sq_err: 65.12924
[ Info:   sum_rel: 119374.03
[ Info:   count_rel: 12472391
[ Info:   scale: 0.007865906
[ Info:   max_abs: 0.0043945312
[ Info:   max_rel: 0.32877603
[ Info:    Макс. абс. погрѣшность: 0.004395
[ Info:    SNR: 48.08 дБ
[ Info:    Средняя отн. погрѣшность: 0.9571%
[ Info:    Макс. отн. погрѣшность (фильтр.): 32.8776%
[ Info: ✅ БЕЗОПАСНО для обученія
[ Info: 🎯 Ожидаемое ускореніе PCIe: ~1.0x
┌ Warning: ⚠️ Пересмотрите результаты передъ интеграціей.
└ @ Main ~/14ext4/ckpt_11l384_cmp50/sm56_defoamer_bench_v2.0.jl:315
[ Info:    Ускореніе можетъ возрасти на большихъ тензорахъ (активацияхъ).
```

```
julia sm56_defoamer_bench_v2.0.jl --help
SM56-Defoamer Benchmark v2.0
Использованіе: julia sm56_defoamer_bench_v2.0.jl [опціи] [чекпоинтъ.jld2]
Опціи:
  -i, --input PATH       Путь къ файлу .jld2
  --min-size-mb SIZE     Минимальный размѣръ тензора въ МБ (по умолчанию 1.0)
  -h, --help             Показать сію справку
Примѣры:
  julia sm56_defoamer_bench_v2.0.jl -i latest.jld2
  julia sm56_defoamer_bench_v2.0.jl --min-size-mb 5.0 latest.jld2
  julia sm56_defoamer_bench_v2.0.jl  (синтетическія данныя)
```
Real checkpoins (no synth):

```
julia sm56_defoamer_bench_v2.0.jl -i ../ckpt_11l384_Qwen37/latest_good.jld2                      
[ Info: 🖥️ GPU: NVIDIA CMP 50HX (SM 7.5, 56 SM)
[ Info: 🧪 SM56-Defoamer Benchmark v2.0
[ Info: 📏 Минимальный размѣръ тензора: 1.0 МБ
[ Info: 📥 Загрузка тензоровъ изъ: ../ckpt_11l384_Qwen37/latest_good.jld2
┌ Warning: type Optimisers.AdamW{Float32,Tuple{Float32, Float32},Float32,Float64} does not exist in workspace; reconstructing
└ @ JLD2 ~/.julia/packages/JLD2/ws7Qu/src/data/reconstructing_datatypes.jl:620
// Warnings are safe and OK (FYI only)

[ Info: ✅ Загружено 112 тензоровъ для испытанія
[ Info: 🔬 Діагностика перваго тензора...
[ Info:   Типъ: CuArray{Float16, 2, CUDA.DeviceMemory}
[ Info:   Размѣръ: (384, 385)
[ Info:   Минимумъ: -2.578
[ Info:   Максимумъ: 2.684
[ Info:   Среднее: -0.011549094
[ Info:   Есть NaN: false
[ Info:   Есть Inf: false
[ Info: 📊 Полезная нагрузка: 28.31 МБ (112 тензоровъ)
[ Info: 🔥 Прогрѣвъ...
[ Info: 📏 Сырой трансферъ FP16
[ Info:    Время: 0.4890 сек | Пропускная способность: 115.8 МБ/сек
[ Info: 📦 Сжатіе SM56-Defoamer INT8
[ Info:    Время: 2.0897 сек | Пропускная способность: 27.1 МБ/сек | Сжатіе: 2.00x | Ускореніе: 0.2x
[ Info: 🔍 Анализъ погрѣшностей (Надежныя метрики)
[ Info:    Макс. абс. погрѣшность: 0.011719
[ Info:    SNR: Inf дБ
[ Info:    Средняя отн. погрѣшность: 1.9566%
[ Info:    Макс. отн. погрѣшность (фильтр.): 33.3740%
[ Info: ⚠️ ПРОВѢРЬТЕ для обученія
[ Info: 🎯 Ожидаемое ускореніе PCIe: ~0.2x
┌ Warning: ⚠️ Пересмотрите результаты передъ интеграціей.
```
