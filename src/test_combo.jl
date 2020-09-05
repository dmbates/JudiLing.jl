"""
test entire dataset in one function
"""
function test_combo end

"""
  test_combo

test entire dataset in one function

...
# Examples
```julia
mkpath(joinpath("french_out"))
test_io = open(joinpath("french_out", "out.log"), "w")

JudiLing.test_combo(
  joinpath("data", "french_mini.csv"),
  joinpath("french_out"),
  ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
  ["Lexeme"],
  ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
  data_prefix="french",
  max_test_data=nothing,
  split_max_ratio=0.1,
  is_full_A=false,
  n_grams_target_col=:Syllables,
  n_grams_tokenized=true,
  n_grams_sep_token="-",
  n_grams_keep_sep=true,
  grams=3,
  start_end_token="#",
  path_sep_token=":",
  learning_mode=:cholesky,
  alpha=0.1,
  betas=(0.1,0.1),
  eta=0.1,
  n_epochs=nothing,
  path_method=:learn_paths,
  max_t=nothing,
  max_can=10,
  train_threshold=0.1,
  val_is_tolerant=false,
  val_threshold=(-100.0),
  val_tolerance=(-1000.0),
  val_max_tolerance=4,
  train_n_neighbors=2,
  val_n_neighbors=10,
  root_dir=@__DIR__,
  csv_dir="french_out",
  csv_prefix="french",
  random_seed=314,
  log_io=test_io,
  verbose=false)

close(test_io)
```
...
"""
function test_combo(
    data_path::String,
    output_dir_path::String,
    n_features_columns::Vector,
    n_features_base::Vector,
    n_features_inflections::Vector;
    data_prefix="data"::String,
    max_test_data=nothing::Union{Nothing, Int64},
    split_max_ratio=0.2::Float64,
    is_full_A=false::Bool,
    n_grams_target_col=:PhonWord::Symbol,
    n_grams_tokenized=false::Bool,
    n_grams_sep_token=""::Union{String, Char},
    n_grams_keep_sep=false::Bool,
    grams=3::Int64,
    start_end_token="#"::Union{String, Char},
    path_sep_token=":"::Union{String, Char},
    learning_mode=:cholesky::Symbol,
    alpha=0.1::Float64,
    betas=(0.1,0.1)::Tuple{Float64,Float64},
    eta=0.1::Float64,
    n_epochs=nothing::Union{Int64, Nothing},
    path_method=:build_paths::Symbol,
    max_t=nothing::Union{Int64, Nothing},
    max_can=10::Int64,
    train_threshold=0.1::Float64,
    val_is_tolerant=false::Bool,
    val_threshold=(-100.0)::Float64,
    val_tolerance=(-1000.0)::Float64,
    val_max_tolerance=4::Int64,
    train_n_neighbors=2::Int64,
    val_n_neighbors=10::Int64,
    root_dir="."::String,
    csv_dir="out"::String,
    csv_prefix="french"::String,
    random_seed=314::Int64,
    log_io=stdout::IO,
    verbose=false::Bool
  )::Nothing

  # split data
  verbose && println("spliting data...")
  train_val_split(
    data_path,
    output_dir_path,
    n_features_columns,
    data_prefix=data_prefix,
    max_test_data=max_test_data,
    split_max_ratio=split_max_ratio,
    n_grams_target_col=n_grams_target_col,
    n_grams_tokenized=n_grams_tokenized,
    n_grams_sep_token=n_grams_sep_token,
    grams=grams,
    n_grams_keep_sep=n_grams_keep_sep,
    start_end_token=start_end_token,
    random_seed=random_seed,
    verbose=verbose
  )

  # load data
  verbose && println("Loading CSV...")
  data_train = CSV.DataFrame!(CSV.File(joinpath(output_dir_path, "$(data_prefix)_train.csv")))
  data_val = CSV.DataFrame!(CSV.File(joinpath(output_dir_path, "$(data_prefix)_val.csv")))

  check_used_token(data_train, n_grams_target_col, start_end_token, "start_end_token")
  check_used_token(data_train, n_grams_target_col, path_sep_token, "path_sep_token")
  check_used_token(data_val, n_grams_target_col, start_end_token, "start_end_token")
  check_used_token(data_val, n_grams_target_col, path_sep_token, "path_sep_token")

  if learning_mode == :cholesky
    verbose && println("Making cue matrix...")
    cue_obj_train = make_cue_matrix(
      data_train,
      grams=grams,
      target_col=n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      verbose=verbose)
    cue_obj_val = make_cue_matrix(
      data_val,
      cue_obj_train,
      grams=grams,
      target_col=n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      verbose=verbose)
    verbose && println("training type: $(typeof(cue_obj_train.C))")
    verbose && println("val type: $(typeof(cue_obj_val.C))")
    verbose && println("training size: $(size(cue_obj_train.C))")
    verbose && println("val size: $(size(cue_obj_val.C))")
    verbose && println()

    n_features = size(cue_obj_train.C, 2)

    verbose && println("Making S matrix...")
    S_train, S_val = make_S_matrix(
      data_train,
      data_val,
      n_features_base,
      n_features_inflections,
      ncol = n_features)
    verbose && println("training type: $(typeof(S_train))")
    verbose && println("val type: $(typeof(S_val))")
    verbose && println("training size: $(size(S_train))")
    verbose && println("val size: $(size(S_val))")
    verbose && println()

    verbose && println("Make G matrix...")
    G_train = make_transform_matrix(S_train, cue_obj_train.C)
    verbose && println("G type: $(typeof(G_train))")
    verbose && println("G size: $(size(G_train))")

    verbose && println("Make F matrix...")
    F_train = make_transform_matrix(cue_obj_train.C, S_train)
    verbose && println("F type: $(typeof(F_train))")
    verbose && println("F size: $(size(F_train))")

  elseif learning_mode == :pyndl
    verbose && println("Preprocessing pyndl text...")
    preprocess_ndl(
      joinpath(output_dir_path, "$(data_prefix)_train.csv"),
      joinpath(output_dir_path, "$(data_prefix)_train.tab.gz"),
      grams=grams,
      n_grams_target_col=n_grams_target_col,
      n_grams_tokenized=n_grams_tokenized,
      n_grams_sep_token=n_grams_sep_token,
      n_grams_keep_sep=n_grams_keep_sep,
      n_features_columns=n_features_columns
      )

    verbose && println("Using pyndl make G matrix...")
    pws = pyndl(
      joinpath(output_dir_path, "$(data_prefix)_train.tab.gz"),
      alpha=alpha,
      betas=betas)

    G_train = pws.weight
    verbose && println("G type: $(typeof(G_train))")
    verbose && println("G size: $(size(G_train))")

    verbose && println("Making cue matrix...")
    cue_obj_train = make_cue_matrix(
      data_train,
      pws,
      grams=grams,
      target_col=n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      verbose=verbose)
    cue_obj_val = make_cue_matrix(
      data_val,
      cue_obj_train,
      grams=grams,
      target_col=n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      verbose=verbose)
    verbose && println("training type: $(typeof(cue_obj_train.C))")
    verbose && println("val type: $(typeof(cue_obj_val.C))")
    verbose && println("training size: $(size(cue_obj_train.C))")
    verbose && println("val size: $(size(cue_obj_val.C))")
    verbose && println()

    n_features = size(cue_obj_train.C, 2)

    verbose && println("Making S matrix...")
    S_train, S_val = make_S_matrix(
      data_train,
      data_val,
      pws,
      n_features_columns)
    verbose && println("training type: $(typeof(S_train))")
    verbose && println("val type: $(typeof(S_val))")
    verbose && println("training size: $(size(S_train))")
    verbose && println("val size: $(size(S_val))")
    verbose && println()

    verbose && println("Make F matrix...")
    F_train = make_transform_matrix(cue_obj_train.C, S_train)
    verbose && println("F type: $(typeof(F_train))")
    verbose && println("F size: $(size(F_train))")
  elseif learning_mode == :wh
    verbose && println("Making cue matrix...")
    cue_obj_train = make_cue_matrix(
      data_train,
      grams=grams,
      target_col=n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      verbose=verbose)
    cue_obj_val = make_cue_matrix(
      data_val,
      cue_obj_train,
      grams=grams,
      target_col=n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      verbose=verbose)
    verbose && println("training type: $(typeof(cue_obj_train.C))")
    verbose && println("val type: $(typeof(cue_obj_val.C))")
    verbose && println("training size: $(size(cue_obj_train.C))")
    verbose && println("val size: $(size(cue_obj_val.C))")
    verbose && println()

    n_features = size(cue_obj_train.C, 2)

    verbose && println("Making S matrix...")
    S_train, S_val = make_S_matrix(
      data_train,
      data_val,
      n_features_base,
      n_features_inflections,
      ncol = n_features)
    verbose && println("training type: $(typeof(S_train))")
    verbose && println("val type: $(typeof(S_val))")
    verbose && println("training size: $(size(S_train))")
    verbose && println("val size: $(size(S_val))")
    verbose && println()

    verbose && println("Using wh make G matrix...")
    G_train = wh_learn(
      cue_obj_train.C,
      S_train,
      eta=eta,
      n_epochs=n_epochs,
      verbose=verbose)
    verbose && println("G type: $(typeof(G_train))")
    verbose && println("G size: $(size(G_train))")

    verbose && println("Make F matrix...")
    F_train = make_transform_matrix(cue_obj_train.C, S_train)
    verbose && println("F type: $(typeof(F_train))")
    verbose && println("F size: $(size(F_train))")

  else
    throw(ArgumentError("learning_mode is incorrect, using :cholesky, :wh or :pyndl"))
  end

  verbose && println("Calculating Chat...")
  Chat_train = convert(Matrix{Float64}, S_train) * G_train
  Chat_val = convert(Matrix{Float64}, S_val) * G_train

  verbose && println("Calculating Shat...")
  Shat_train = convert(Matrix{Float64}, cue_obj_train.C) * F_train
  Shat_val = convert(Matrix{Float64}, cue_obj_val.C) * F_train

  verbose && println("Calculating A...")
  if is_full_A
    A = make_adjacency_matrix(
      cue_obj_train.i2f,
      tokenized = n_grams_tokenized,
      sep_token = n_grams_sep_token,
      verbose=verbose)
  else
    A = cue_obj_train.A
  end

  if isnothing(max_t)
    max_t = cal_max_timestep(
      data_train,
      data_val,
      n_grams_target_col,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token)
  end

  verbose && println("Finding paths...")
  if path_method==:learn_paths
    res_train, gpi_train = learn_paths(
      data_train,
      data_train,
      cue_obj_train.C,
      S_train,
      F_train,
      Chat_train,
      A,
      cue_obj_train.i2f,
      gold_ind=cue_obj_train.gold_ind,
      Shat_val=Shat_train,
      check_gold_path=true,
      max_t=max_t,
      max_can=max_can,
      grams=grams,
      threshold=train_threshold,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      target_col=n_grams_target_col,
      issparse=:dense,
      verbose=verbose)

    res_val, gpi_val = learn_paths(
      data_train,
      data_val,
      cue_obj_train.C,
      S_val,
      F_train,
      Chat_val,
      A,
      cue_obj_train.i2f,
      gold_ind=cue_obj_val.gold_ind,
      Shat_val=Shat_val,
      check_gold_path=true,
      max_t=max_t,
      max_can=max_can,
      grams=grams,
      threshold=val_threshold,
      is_tolerant=val_is_tolerant,
      tolerance=val_tolerance,
      max_tolerance=val_max_tolerance,
      tokenized=n_grams_tokenized,
      sep_token=n_grams_sep_token,
      keep_sep=n_grams_keep_sep,
      target_col=n_grams_target_col,
      issparse=:dense,
      verbose=verbose)
  else
    res_train = build_paths(
      data_train,
      cue_obj_train.C,
      S_train,
      F_train,
      Chat_train,
      A,
      cue_obj_train.i2f,
      cue_obj_train.gold_ind,
      max_t=max_t,
      n_neighbors=train_n_neighbors,
      verbose=verbose
      )

    res_val = build_paths(
      data_val,
      cue_obj_train.C,
      S_val,
      F_train,
      Chat_val,
      A,
      cue_obj_train.i2f,
      cue_obj_train.gold_ind,
      max_t=max_t,
      n_neighbors=val_n_neighbors,
      verbose=verbose
      )
  end

  verbose && println("Evaluate acc...")
  acc_train = eval_acc(
    res_train,
    cue_obj_train.gold_ind,
    verbose=verbose
  )
  acc_val = eval_acc(
    res_val,
    cue_obj_val.gold_ind,
    verbose=verbose
  )
  acc_train_loose = eval_acc_loose(
    res_train,
    cue_obj_train.gold_ind,
    verbose=verbose
  )
  acc_val_loose = eval_acc_loose(
    res_val,
    cue_obj_val.gold_ind,
    verbose=verbose
  )
  println(log_io, "Acc for train: $acc_train")
  println(log_io, "Acc for val: $acc_val")
  println(log_io, "Acc for train loose: $acc_train_loose")
  println(log_io, "Acc for val loose: $acc_val_loose")

  write2csv(
    res_train,
    data_train,
    cue_obj_train,
    cue_obj_train,
    "res_$(csv_prefix)_train.csv",
    grams=grams,
    tokenized=n_grams_tokenized,
    sep_token=n_grams_sep_token,
    start_end_token=start_end_token,
    output_sep_token=n_grams_sep_token,
    path_sep_token=path_sep_token,
    root_dir=root_dir,
    output_dir=csv_dir,
    target_col=n_grams_target_col)

  write2csv(
    res_val,
    data_val,
    cue_obj_train,
    cue_obj_val,
    "res_$(csv_prefix)_val.csv",
    grams=grams,
    tokenized=n_grams_tokenized,
    sep_token=n_grams_sep_token,
    start_end_token=start_end_token,
    output_sep_token=n_grams_sep_token,
    path_sep_token=path_sep_token,
    root_dir=root_dir,
    output_dir=csv_dir,
    target_col=n_grams_target_col)

  nothing
end