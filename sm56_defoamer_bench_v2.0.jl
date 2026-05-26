#!/usr/bin/env julia
# SM56-Defoamer Benchmark v2.0: Исправленіе переполненія Float16 и добавленіе діагностики
# Fixes: Inf SNR/MeanRel due to Float16 overflow on large tensors.
# Metrics now computed in Float32 for safety and precision.
# Debug output added for first tensor.

using CUDA, JLD2, Printf, Statistics

# ============================================================
# 1. Разборъ аргументовъ
# ============================================================
function parse_args()
    path = ""
    min_size_mb = 1.0
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "-i" || ARGS[i] == "--input"
            i += 1
            if i <= length(ARGS)
                path = ARGS[i]
            end
        elseif ARGS[i] == "--min-size-mb"
            i += 1
            if i <= length(ARGS)
                min_size_mb = parse(Float64, ARGS[i])
            end
        elseif ARGS[i] == "-h" || ARGS[i] == "--help"
            println("SM56-Defoamer Benchmark v2.0")
            println("Использованіе: julia sm56_defoamer_bench_v2.0.jl [опціи] [чекпоинтъ.jld2]")
            println("Опціи:")
            println("  -i, --input PATH       Путь къ файлу .jld2")
            println("  --min-size-mb SIZE     Минимальный размѣръ тензора въ МБ (по умолчанию 1.0)")
            println("  -h, --help             Показать сію справку")
            println("Примѣры:")
            println("  julia sm56_defoamer_bench_v2.0.jl -i latest.jld2")
            println("  julia sm56_defoamer_bench_v2.0.jl --min-size-mb 5.0 latest.jld2")
            println("  julia sm56_defoamer_bench_v2.0.jl  (синтетическія данныя)")
            exit(0)
        elseif !startswith(ARGS[i], "-")
            path = ARGS[i]
        end
        i += 1
    end
    return path, min_size_mb
end

# ============================================================
# 2. Опредѣленіе GPU
# ============================================================
function detect_gpu_info()
    dev = CUDA.device()
    name = CUDA.name(dev)
    cap = CUDA.capability(dev)
    major = Int(cap.major)
    minor = Int(cap.minor)
    sm_count = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    return name, major, minor, sm_count
end

# ============================================================
# 3. SM56-Defoamer: Исправленная реализація
# ============================================================

function sm56_compress(x::CuArray{Float16})
    n = length(x)
    if n == 0
        return Int8[], Float16(0.0)
    end
    
    x_abs_f32 = Float32.(abs.(x))
    m = maximum(x_abs_f32)
    
    if !isfinite(m) || m <= 0f0
        m = 1f-8
    end
    
    scale = Float16(m / 127.0f0)
    q = Int8.(clamp.(round.(x ./ scale), -127, 127))
    
    return Array(q), scale
end

function sm56_decompress(data::Array{Int8}, scale::Float16, shape::Dims)
    q = CuArray(data)
    return reshape(Float16.(q) .* scale, shape)
end

# ============================================================
# 4. Загрузка данныхъ
# ============================================================
function load_tensors(path::String; min_size_mb::Float64=1.0)
    @info "📥 Загрузка тензоровъ изъ: $path"
    ck = JLD2.load(path)
    all_tensors = CuArray{Float16}[]
    
    function collect!(obj)
        if obj isa AbstractArray && eltype(obj) <: AbstractFloat
            push!(all_tensors, CuArray{Float16}(obj))
        elseif obj isa Union{NamedTuple, Tuple, Dict}
            for v in values(obj); collect!(v); end
        end
    end
    
    for v in values(ck); collect!(v); end
    
    min_bytes = min_size_mb * 1024^2
    tensors = filter(t -> sizeof(t) >= min_bytes, all_tensors)
    filtered_count = length(all_tensors) - length(tensors)
    
    if filtered_count > 0
        @info "🔍 Исключено $filtered_count малыхъ тензоровъ (< $(min_size_mb) МБ)"
    end
    
    if isempty(tensors)
        @warn "⚠️ Нѣтъ тензоровъ ≥ $(min_size_mb) МБ. Попробуйте уменьшить порогъ."
    else
        @info "✅ Загружено $(length(tensors)) тензоровъ для испытанія"
    end
    return tensors
end

# ============================================================
# 5. Синтетическія данныя
# ============================================================
function make_synthetic_v2(; min_size_mb::Float64=1.0)
    @info "🎲 Генерація синтетическихъ активаций (увеличенный размѣръ)"
    
    dim = 384
    seq = 512
    batch = 64
    
    tensor_bytes = dim * seq * batch * 2
    tensor_mb = tensor_bytes / 1024^2
    
    @info "  Размѣръ одного тензора: $(round(tensor_mb, digits=2)) МБ"
    
    if tensor_mb < min_size_mb
        required_batch = ceil(Int, min_size_mb * 1024^2 / (dim * seq * 2))
        batch = required_batch
        tensor_bytes = dim * seq * batch * 2
        tensor_mb = tensor_bytes / 1024^2
        @info "  ⚠️ Batch увеличенъ до $batch для достиженія порога $(min_size_mb) МБ"
    end
    
    tensors = [CUDA.rand(Float16, dim, seq, batch) for _ in 1:22]
    
    total_mb = sum(sizeof(t) for t in tensors) / 1024^2
    @info "  ✅ Сгенерировано $(length(tensors)) тензоровъ, общій объемъ: $(round(total_mb, digits=2)) МБ"
    
    return tensors
end

# ============================================================
# 6. Діагностика
# ============================================================
function run_diagnostics(tensors)
    @info "🔬 Діагностика перваго тензора..."
    x = tensors[1]
    x_cpu = Array(x)
    @info "  Типъ: $(typeof(x))"
    @info "  Размѣръ: $(size(x))"
    @info "  Минимумъ: $(minimum(x_cpu))"
    @info "  Максимумъ: $(maximum(x_cpu))"
    @info "  Среднее: $(mean(Float32.(x_cpu)))"
    @info "  Есть NaN: $(any(isnan, x_cpu))"
    @info "  Есть Inf: $(any(isinf, x_cpu))"
end

# ============================================================
# 7. Ядро испытанія
# ============================================================
function run_benchmark(tensors)
    total_bytes = sum(sizeof(t) for t in tensors)
    @info @sprintf("📊 Полезная нагрузка: %.2f МБ (%d тензоровъ)", total_bytes/1024^2, length(tensors))
    
    if total_bytes < 10 * 1024^2
        @warn "⚠️ Нагрузка слишкомъ мала (<10 МБ). Результаты могутъ быть неточны."
    end
    
    CUDA.synchronize(); GC.gc(false); CUDA.reclaim()

    @info "🔥 Прогрѣвъ..."
    _ = [Array(t) for t in tensors]
    _ = [sm56_compress(t) for t in tensors]
    CUDA.synchronize()

    @info "📏 Сырой трансферъ FP16"
    t0 = time_ns()
    raw_cpu = [Array(t) for t in tensors]
    CUDA.synchronize()
    t1 = time_ns()
    _ = [CuArray(d) for d in raw_cpu]
    CUDA.synchronize()
    t2 = time_ns()
    
    dt_raw = (t2 - t0) / 1e9
    bw_raw = (total_bytes * 2) / dt_raw / 1024^2
    @info @sprintf("   Время: %.4f сек | Пропускная способность: %.1f МБ/сек", dt_raw, bw_raw)

    @info "📦 Сжатіе SM56-Defoamer INT8"
    t0 = time_ns()
    comp_data = [sm56_compress(t) for t in tensors]
    CUDA.synchronize()
    t1 = time_ns()
    decomp_gpu = [sm56_decompress(d[1], d[2], size(tensors[i])) for (i,d) in enumerate(comp_data)]
    CUDA.synchronize()
    t2 = time_ns()
    
    dt_comp = (t2 - t0) / 1e9
    compact_bytes = sum(sizeof(d[1]) + sizeof(d[2]) for d in comp_data)
    bw_comp = (total_bytes * 2) / dt_comp / 1024^2
    ratio = compact_bytes / total_bytes
    speedup = dt_raw / dt_comp
    
    @info @sprintf("   Время: %.4f сек | Пропускная способность: %.1f МБ/сек | Сжатіе: %.2fx | Ускореніе: %.1fx",
                   dt_comp, bw_comp, 1/ratio, speedup)

    @info "🔍 Анализъ погрѣшностей (Float32 метрики, защита отъ переполненія)"
    
    max_abs = 0.0f0
    sum_sq_err = 0.0f0
    sum_sq_sig = 0.0f0
    sum_rel = 0.0f0
    count_rel = 0
    max_rel_filtered = 0.0f0
    
    for i in 1:length(tensors)
        # Переводимъ въ Float32 для вычисленій, дабы избѣжать переполненія Float16
        orig = Float32.(Array(tensors[i]))
        rec = Float32.(Array(decomp_gpu[i]))
        
        err = abs.(orig .- rec)
        max_abs = max(max_abs, maximum(err))
        
        # Накопленіе суммъ въ Float32
        sum_sq_err += sum(err.^2)
        sum_sq_sig += sum(orig.^2)
        
        scale = Float32(comp_data[i][2])
        mask = abs.(orig) .> max(scale, 1f-6)
        
        if any(mask)
            rel = err[mask] ./ abs.(orig[mask])
            sum_rel += sum(rel)
            count_rel += count(mask)
            max_rel_filtered = max(max_rel_filtered, maximum(rel))
        end
        
        # Діагностика для перваго тензора
        if i == 1
            @info "🐛 Діагностика метрикъ (первый тензоръ):"
            @info "  sum_sq_sig: $(sum(orig.^2))"
            @info "  sum_sq_err: $(sum(err.^2))"
            @info "  sum_rel: $(sum(rel))"
            @info "  count_rel: $(count(mask))"
            @info "  scale: $scale"
            @info "  max_abs: $(maximum(err))"
            @info "  max_rel: $(maximum(rel))"
        end
    end
    
    # Вычисленіе финальныхъ метрикъ
    snr_db = 10 * log10(sum_sq_sig / (sum_sq_err + 1f-12))
    mean_rel = count_rel > 0 ? (sum_rel / count_rel) : 0.0f0
    
    @info @sprintf("   Макс. абс. погрѣшность: %.6f", max_abs)
    @info @sprintf("   SNR: %.2f дБ", snr_db)
    @info @sprintf("   Средняя отн. погрѣшность: %.4f%%", mean_rel * 100)
    @info @sprintf("   Макс. отн. погрѣшность (фильтр.): %.4f%%", max_rel_filtered * 100)
    
    safe = snr_db > 45.0 && mean_rel < 0.01
    status = safe ? "✅ БЕЗОПАСНО" : "⚠️ ПРОВѢРЬТЕ"
    @info "$status для обученія"
    
    expected_speedup = speedup * 0.85
    @info @sprintf("🎯 Ожидаемое ускореніе PCIe: ~%.1fx", expected_speedup)
    
    return safe, expected_speedup
end

# ============================================================
# 8. Главная функція
# ============================================================
function main()
    path, min_size_mb = parse_args()
    
    name, major, minor, sm = detect_gpu_info()
    @info "🖥️ GPU: $name (SM $major.$minor, $sm SM)"
    @info "🧪 SM56-Defoamer Benchmark v2.0"
    @info "📏 Минимальный размѣръ тензора: $(min_size_mb) МБ"
    
    tensors = if !isempty(path) && isfile(path)
        load_tensors(path; min_size_mb=min_size_mb)
    else
        if !isempty(path)
            @warn "Файлъ не найденъ: $path. Используются синтетическія данныя."
        end
        make_synthetic_v2(min_size_mb=min_size_mb)
    end
    
    if isempty(tensors)
        @error "Нѣтъ тензоровъ для испытанія. Попробуйте уменьшить --min-size-mb."
        return
    end
    
    run_diagnostics(tensors)
    
    safe, speedup = run_benchmark(tensors)
    
    if safe && speedup > 1.5
        @info "🚀 SM56-Defoamer готовъ къ производству."
        @info "   Интегрируйте sm56_compress/sm56_decompress въ hybrid_checkpoint."
        @info "   Примѣчаніе: Пиковая VRAM = 2.0× входъ изъ-за буфера FP32."
    else
        @warn "⚠️ Пересмотрите результаты передъ интеграціей."
        if speedup <= 1.5
            @info "   Ускореніе можетъ возрасти на большихъ тензорахъ (активацияхъ)."
        end
    end
end

main()
