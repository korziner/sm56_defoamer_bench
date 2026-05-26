#!/usr/bin/env julia
# SM56-Defoamer Benchmark v3.1: Memory-Safe INT8 vs TQ3_1S
# Fixes: OOM errors by processing tensors sequentially and freeing memory.
# Вычисляетъ метрики "на лету", не накапливая возстановленные тензоры.
# Дореформенная орѳографія.

using CUDA, JLD2, Printf, Statistics

# Подключеніе модуля TQ3_1S
include("TQ3_1S.jl")
using .TQ3_1S

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
            if i <= length(ARGS) path = ARGS[i] end
        elseif ARGS[i] == "--min-size-mb"
            i += 1
            if i <= length(ARGS) min_size_mb = parse(Float64, ARGS[i]) end
        elseif ARGS[i] == "-h" || ARGS[i] == "--help"
            println("SM56-Defoamer Benchmark v3.1 (Memory-Safe)")
            println("Использованіе: julia sm56_defoamer_bench_v3.1.jl [опціи] [чекпоинтъ.jld2]")
            exit(0)
        elseif !startswith(ARGS[i], "-")
            path = ARGS[i]
        end
        i += 1
    end
    return path, min_size_mb
end

# ============================================================
# 2. Компрессоры
# ============================================================
function sm56_compress(x::CuArray{Float16})
    n = length(x)
    if n == 0 return CuArray{Int8}(), Float16(0.0), size(x) end
    x_abs_f32 = Float32.(abs.(x))
    m = maximum(x_abs_f32)
    if !isfinite(m) || m <= 0f0 m = 1f-8 end
    scale = Float16(m / 127.0f0)
    q = Int8.(clamp.(round.(x ./ scale), -127, 127))
    return q, scale, size(x)
end

function sm56_decompress(q::CuArray{Int8}, scale::Float16, shape::Dims)
    return reshape(Float16.(q) .* scale, shape)
end

function tq3_compress(x::CuArray{Float16})
    data, scales, shape = TQ3_1S.tq3_1s_compress(x)
    return data, scales, shape
end

function tq3_decompress(data::CuArray{UInt8}, scales::CuArray{Float16}, shape::Dims)
    return TQ3_1S.tq3_1s_decompress(data, scales, shape)
end

# ============================================================
# 3. Загрузка данныхъ
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
    if isempty(tensors)
        @warn "⚠️ Нѣтъ тензоровъ ≥ $(min_size_mb) МБ."
    else
        @info "✅ Загружено $(length(tensors)) тензоровъ"
    end
    return tensors
end

function make_synthetic()
    @info "🎲 Генерація синтетическихъ активаций"
    # 384 * 512 * 32 * 2 bytes = 12.58 MB per tensor.
    # 22 tensors = ~276 MB total. Safe for 10GB VRAM.
    return [CUDA.rand(Float16, 384, 512, 32) for _ in 1:22]
end

# ============================================================
# 4. Ядро испытанія (Memory-Safe)
# ============================================================
function run_benchmark(tensors)
    total_bytes = sum(sizeof(t) for t in tensors)
    @info @sprintf("📊 Полезная нагрузка: %.2f МБ (%d тензоровъ)", total_bytes/1024^2, length(tensors))
    
    CUDA.synchronize(); GC.gc(false); CUDA.reclaim()
    @info "🔥 Прогрѣвъ..."
    _ = sm56_compress(tensors[1])
    _ = tq3_compress(tensors[1])
    CUDA.synchronize()
    
    # Массивы для накопленія временъ
    raw_times = zeros(length(tensors))
    int8_compute = zeros(length(tensors))
    int8_transfer = zeros(length(tensors))
    tq3_compute = zeros(length(tensors))
    tq3_transfer = zeros(length(tensors))
    
    # Метрики точности
    int8_sum_sq_err = 0f0; int8_sum_sq_sig = 0f0; int8_max_abs = 0f0
    tq3_sum_sq_err = 0f0; tq3_sum_sq_sig = 0f0; tq3_max_abs = 0f0
    
    @info "🔄 Послѣдовательная обработка тензоровъ (Memory-Safe)..."
    
    for i in 1:length(tensors)
        t = tensors[i]
        
        # === RAW ===
        CUDA.synchronize()
        t0 = time_ns()
        cpu_t = Array(t)
        CUDA.synchronize()
        t1 = time_ns()
        gpu_t = CuArray(cpu_t)
        CUDA.synchronize()
        t2 = time_ns()
        raw_times[i] = (t2 - t0) / 1e9
        
        # === INT8 ===
        CUDA.synchronize()
        t0 = time_ns()
        q, scale, shape = sm56_compress(t)
        CUDA.synchronize()
        tc1 = (time_ns() - t0) / 1e9
        
        t0 = time_ns()
        cpu_q = Array(q)
        CUDA.synchronize()
        tt1 = (time_ns() - t0) / 1e9
        
        t0 = time_ns()
        gpu_q = CuArray(cpu_q)
        CUDA.synchronize()
        tt2 = (time_ns() - t0) / 1e9
        
        t0 = time_ns()
        rec_int8 = sm56_decompress(gpu_q, scale, shape)
        CUDA.synchronize()
        tc2 = (time_ns() - t0) / 1e9
        
        int8_compute[i] = tc1 + tc2
        int8_transfer[i] = tt1 + tt2
        
        # Метрики INT8 (на CPU, затѣмъ освобождаемъ GPU)
        orig_f32 = Float32.(cpu_t)
        rec_f32 = Float32.(Array(rec_int8))
        err = abs.(orig_f32 .- rec_f32)
        int8_max_abs = max(int8_max_abs, maximum(err))
        int8_sum_sq_err += sum(err.^2)
        int8_sum_sq_sig += sum(orig_f32.^2)
        
        gpu_q = nothing; rec_int8 = nothing # Освобождаемъ VRAM
        
        # === TQ3_1S ===
        CUDA.synchronize()
        t0 = time_ns()
        data, scales, shape = tq3_compress(t)
        CUDA.synchronize()
        tc1 = (time_ns() - t0) / 1e9
        
        t0 = time_ns()
        cpu_data = Array(data)
        cpu_scales = Array(scales)
        CUDA.synchronize()
        tt1 = (time_ns() - t0) / 1e9
        
        t0 = time_ns()
        gpu_data = CuArray(cpu_data)
        gpu_scales = CuArray(cpu_scales)
        CUDA.synchronize()
        tt2 = (time_ns() - t0) / 1e9
        
        t0 = time_ns()
        rec_tq3 = tq3_decompress(gpu_data, gpu_scales, shape)
        CUDA.synchronize()
        tc2 = (time_ns() - t0) / 1e9
        
        tq3_compute[i] = tc1 + tc2
        tq3_transfer[i] = tt1 + tt2
        
        # Метрики TQ3_1S
        rec_f32_tq = Float32.(Array(rec_tq3))
        err_tq = abs.(orig_f32 .- rec_f32_tq)
        tq3_max_abs = max(tq3_max_abs, maximum(err_tq))
        tq3_sum_sq_err += sum(err_tq.^2)
        tq3_sum_sq_sig += sum(orig_f32.^2)
        
        gpu_data = nothing; gpu_scales = nothing; rec_tq3 = nothing # Освобождаемъ VRAM
        
        GC.gc(false); CUDA.reclaim()
    end
    
    # ============================================================
    # 5. Агрегация и выводъ результатовъ
    # ============================================================
    dt_raw = sum(raw_times)
    dt_int8_comp = sum(int8_compute)
    dt_int8_trans = sum(int8_transfer)
    dt_tq3_comp = sum(tq3_compute)
    dt_tq3_trans = sum(tq3_transfer)
    
    dt_int8_total = dt_int8_comp + dt_int8_trans
    dt_tq3_total = dt_tq3_comp + dt_tq3_trans
    
    pcie_int8 = dt_raw / dt_int8_trans
    pcie_tq3 = dt_raw / dt_tq3_trans
    e2e_int8 = dt_raw / dt_int8_total
    e2e_tq3 = dt_raw / dt_tq3_total
    
    int8_snr = 10 * log10(int8_sum_sq_sig / (int8_sum_sq_err + 1f-12))
    tq3_snr = 10 * log10(tq3_sum_sq_sig / (tq3_sum_sq_err + 1f-12))
    
    println("\n" * "="^90)
    println(@sprintf("%-15s | %-12s | %-12s | %-12s | %-12s", 
                     "Методъ", "PCIe Speedup ", "E2E Speedup  ", "Compute %", "SNR дБ"))
    println("-"^90)
    println(@sprintf("%-15s | %-12s | %-12s | %-12s | %-12s", 
                     "RAW FP16", "1.00        x", "1.00        x", "0.0%", "∞"))
    println(@sprintf("%-15s | %-12.2fx | %-12.2fx | %-12.1f | %-12.2f", 
                     "INT8", pcie_int8, e2e_int8, (dt_int8_comp/dt_raw)*100, int8_snr))
    println(@sprintf("%-15s | %-12.2fx | %-12.2fx | %-12.1f | %-12.2f", 
                     "TQ3_1S", pcie_tq3, e2e_tq3, (dt_tq3_comp/dt_raw)*100, tq3_snr))
    println("="^90)
    println(@sprintf("%-15s | %-15s | %-15s", "Методъ", "Max Abs Err", "Сжатіе"))
    println("-"^90)
    println(@sprintf("%-15s | %-15.6f | %-15.2fx", "INT8", int8_max_abs, 2.0))
    println(@sprintf("%-15s | %-15.6f | %-15.2fx", "TQ3_1S", tq3_max_abs, TQ3_1S.TQ3_1S_Ratio))
    println("="^90 * "\n")
    
    # Рекомендація
    best_method = e2e_tq3 > e2e_int8 && tq3_snr > 40.0 ? :TQ3_1S : :INT8
    best_score = best_method == :TQ3_1S ? e2e_tq3 : e2e_int8
    
    if best_score > 1.2
        @info @sprintf("🚀 Рекомендованъ методъ: %s съ E2E Speedup   %.2fx", best_method, best_score)
    else
        @warn @sprintf("⚠️ Ускореніе умѣренное (%.2fx). Проверьте загрузку PCIe.", best_score)
    end
end

# ============================================================
# 6. Главная функція
# ============================================================
function main()
    path, min_size_mb = parse_args()
    name = CUDA.name(CUDA.device())
    @info "🖥️ GPU: $name"
    @info "🧪 SM56-Defoamer Benchmark v3.1 (Memory-Safe)"
    
    tensors = if !isempty(path) && isfile(path)
        load_tensors(path; min_size_mb=min_size_mb)
    else
        make_synthetic()
    end
    
    isempty(tensors) && return
    run_benchmark(tensors)
end

main()
