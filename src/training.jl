function loss_comet(model, s, s_dot_true)
    s_dot_pred, s_dot_0, c = model(s)
    loss_dyn = Flux.mse(s_dot_pred, s_dot_true)
    loss_s0 = Flux.mse(s_dot_0, s_dot_true)

    # Regularization term with noisy states [cite: 1650]
    s_noisy = s .+ 0.1f0 .* randn(Float32, size(s))
    if model.ncom >0
        ∇c_noisy = Zygote.jacobian(x -> model.nn(x)[model.nstates+1:end], s_noisy)[1]
        s_dot_0_noisy = model.nn(s_noisy)[1:model.nstates,:]
        loss_reg = sum(abs2, ∇c_noisy' * s_dot_0_noisy)
    else
        loss_reg=0.0f0
    end

    return loss_dyn + loss_s0 + loss_reg
end


function loss_metacomet_phase1(model, s)
    s_dot_0 = model.s_dot_path(s)

    # Orthogonality constraint [cite: 116]
    s_noisy = s .+ 0.1f0 .* randn(Float32, size(s))
    ∇c_noisy = Zygote.jacobian(x -> model.c_path(x), s_noisy)[1]
    s_dot_0_noisy = model.s_dot_path(s_noisy)
    loss_ortho = sum(abs2, ∇c_noisy' * s_dot_0_noisy)


    # SVD semi-orthogonal constraints [cite: 120, 122]
    loss_svd = 0.0f0
    for layer_params in model.shared_layers
        loss_svd += Flux.mse(layer_params.S' * layer_params.S, I) + Flux.mse(layer_params.D' * layer_params.D, I)
    end

    return loss_ortho + loss_svd
end

function loss_metacomet_phase2(model, s, s_dot_true)
    s_dot_pred, s_dot_0, _ = model(s)
    loss_dyn = Flux.mse(s_dot_pred, s_dot_true)
    loss_s0 = Flux.mse(s_dot_0, s_dot_true)
    return loss_dyn + loss_s0  # [cite: 173]
end

function train!(model::UnifiedModel, data_loader, epochs::Int; use_gpu::Bool=false)
    if use_gpu
        CUDA.functional() || error("CUDA is not available. Please check your setup.")
        model = model |> gpu
    end
    opt = ADAM(3e-4)

    if model.model isa MetaCOMETModel
        # --- Phase 1 Training for MetaCOMET ---
        @info "Starting MetaCOMET Training - Phase 1"
        ps_phase1 = Flux.params(model)
        for epoch in 1:epochs
            p_bar = Progress(length(data_loader); desc="Epoch $epoch (Phase 1): ")
            for (s, _) in data_loader
                s_dev = use_gpu ? s |> gpu : s
                gs = gradient(() -> loss_metacomet_phase1(model.model, s_dev), ps_phase1)
                Flux.update!(opt, ps_phase1, gs)
                next!(p_bar)
            end
        end

        # --- Phase 2 Training for MetaCOMET ---
        @info "Starting MetaCOMET Training - Phase 2"
        # Parameters for phase 2 (excluding S and D matrices) [cite: 126, 172]
        ps_phase2 = Flux.params(
            model.model.s_dot_path[1], model.model.s_dot_path[end],
            model.model.c_path[1], model.model.c_path[end],
            [p.V for p in model.model.shared_layers]...
        )
        # Re-initialize the weights of trainable parameters for Phase 2
        for p in ps_phase2
            p .= randn(Float32, size(p)) .* 0.01f0
        end

        for epoch in 1:epochs
            p_bar = Progress(length(data_loader); desc="Epoch $epoch (Phase 2): ")
            for (s, s_dot_true) in data_loader
                s_dev = use_gpu ? s |> gpu : s
                s_dot_true_dev = use_gpu ? s_dot_true |> gpu : s_dot_true
                gs = gradient(() -> loss_metacomet_phase2(model.model, s_dev, s_dot_true_dev), ps_phase2)
                Flux.update!(opt, ps_phase2, gs)
                next!(p_bar)
            end
        end

    else # Standard COMET training
        @info "Starting COMET Training"
        ps = Flux.params(model)
        for epoch in 1:epochs
            p_bar = Progress(length(data_loader); desc="Epoch $epoch: ")
            for (s, s_dot_true) in data_loader
                s_dev = use_gpu ? s |> gpu : s
                s_dot_true_dev = use_gpu ? s_dot_true |> gpu : s_dot_true
                gs = gradient(() -> loss_comet(model.model, s_dev, s_dot_true_dev), ps)
                Flux.update!(opt, ps, gs)
                next!(p_bar)
            end
        end
    end
    return model
end
