using JudiLing
using CSV
using Test

@testset "make prelinguistic semantic matrix for utterance" begin
  try
    utterance = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "utterance_mini.csv")))
    s_obj_train = JudiLing.make_pS_matrix(utterance)

    utterance_val = utterance[101:end, :]
    s_obj_val = JudiLing.make_pS_matrix(utterance_val, s_obj_train)

    @test true
  catch e
    @test e == false
  end
end

@testset "make semantic matrix for french" begin
  try
    french = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "french_mini.csv")))
    S_train = JudiLing.make_S_matrix(
      french,
      ["Lexeme"],
      ["Tense","Aspect","Person","Number","Gender","Class","Mood"])

    french_val = french[100:end,:]
    S_train, S_val = JudiLing.make_S_matrix(
      french,
      french_val,
      ["Lexeme"],
      ["Tense","Aspect","Person","Number","Gender","Class","Mood"])

    S_train = JudiLing.make_S_matrix(
      french,
      ["Lexeme"])

    S_train, S_val = JudiLing.make_S_matrix(
      french,
      french_val,
      ["Lexeme"])
    @test true
  catch e
    @test e == false
  end
end

@testset "make semantic matrix for lexome" begin
  try
    latin = CSV.DataFrame!(CSV.File(joinpath(@__DIR__, "data", "latin_mini.csv")))
    latin_val = latin[1:20,:]

    L1 = JudiLing.make_L_matrix(
      latin,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      ncol=20)

    L2 = JudiLing.make_L_matrix(
      latin,
      ["Lexeme"],
      ncol=20)

    S1 = JudiLing.make_S_matrix(
      latin,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    S1 = JudiLing.make_S_matrix(
      latin,
      ["Lexeme"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    S1, S2 = JudiLing.make_S_matrix(
      latin,
      latin_val,
      ["Lexeme"],
      ["Person","Number","Tense","Voice","Mood"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    S1, S2 = JudiLing.make_S_matrix(
      latin,
      latin_val,
      ["Lexeme"],
      L1,
      add_noise=true,
      sd_noise=1,
      normalized=false
      )

    @test true
  catch e
    @test e == false
  end
end