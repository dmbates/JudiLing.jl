using JuLDL
using CSV
using Test

@testset "make prelinguistic semantic matrix for utterance" begin
  try
    utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
    s_obj_train = JuLDL.make_pS_matrix(utterance)

    utterance_val = utterance[101:end, :]
    s_obj_val = JuLDL.make_pS_matrix(utterance_val, s_obj_train)

    @test true
  catch e
    @test false
  end
end

@testset "make semantic matrix for french" begin
  try
    french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
    S_train = JuLDL.make_S_matrix(
      french,
      ["Lexeme"],
      ["Tense","Aspect","Person","Number","Gender","Class","Mood"])

    french_val = french[100:end,:]
    S_train, S_val = JuLDL.make_S_matrix(
      french,
      french_val,
      ["Lexeme"],
      ["Tense","Aspect","Person","Number","Gender","Class","Mood"])

    S_train = JuLDL.make_S_matrix(
      french,
      base=["Lexeme"])

    S_train, S_val = JuLDL.make_S_matrix(
      french,
      french_val,
      base=["Lexeme"])
    @test true
  catch e
    @test false
  end
end