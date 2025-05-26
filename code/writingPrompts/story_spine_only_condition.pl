% increase the stack limit
:- set_prolog_flag(stack_limit, 8000000000).

% Probs is just for calculating the perplexity

start_story(H0, C0, X0, H6, C6, X6, Probs) --> [X0], generate_part(story_spine(1), H0, C0, X0, H1, C1, X1, Probs1), generate_part(story_spine(2), H1, C1, X1, H2, C2, X2, Probs2), generate_part(story_spine(3), H2, C2, X2, H3, C3, X3, Probs3), generate_part(story_spine(4), H3, C3, X3, H4, C4, X4, Probs4), generate_part(story_spine(5), H4, C4, X4, H5, C5, X5, Probs5), generate_part(story_spine(6), H5, C5, X5, H6, C6, X6, Probs6), {append([Probs1, Probs2, Probs3, Probs4, Probs5, Probs6], Probs)}.


generate_part(story_spine(1), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 0, Hout, Cout, Xout, Probs).
generate_part(story_spine(2), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 0.2, Hout, Cout, Xout, Probs).
generate_part(story_spine(3), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, -0.8, Hout, Cout, Xout, Probs).

% Part 4 can be repeated
generate_part(story_spine(4), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> {generate_condition(Random)}, lstm(Hin, Cin, Xin, Random, Hint, Cint, Xint, Probs1), {random(R)}, test_for_repetition(story_spine(4), R, Hint, Cint, Xint, Hout, Cout, Xout, Probs2), {append([Probs1, Probs2], Probs)}.

generate_part(story_spine(5), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 0.8, Hout, Cout, Xout, Probs).
generate_part(story_spine(6), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 0.8, Hout, Cout, Xout, Probs).


test_for_repetition(story_spine(4), R, Hout, Cout, Xout, Hout, Cout, Xout, []) --> {R =< 0.5}, [].
test_for_repetition(story_spine(4), R, Hint, Cint, Xint, Hout, Cout, Xout, Probs) --> {R > 0.5}, generate_part(story_spine(4), Hint, Cint, Xint, Hout, Cout, Xout, Probs).


lstm(Hin, Cin, Xin, Cond, Hout, Cout, Xout, Probs) --> {py_call(use_sentiment_writingprompts_grammar_only_condition:generate(Hin, Cin, Xin, Cond), -(Hout, Cout, Xout, Generated, Probs))}, Generated.


% useful predicates
generate_condition(Random) :- 
    random(R), 
    Random is 2 * (R - 1/2).