% increase the stack limit
:- set_prolog_flag(stack_limit, 8000000000).

% Probs is just for calculating the perplexity

start_story(H0, C0, H6, C6, Probs) --> generate_part(story_spine(1), H0, C0, H1, C1, Probs1), generate_part(story_spine(2), H1, C1, H2, C2, Probs2), generate_part(story_spine(3), H2, C2, H3, C3, Probs3), generate_part(story_spine(4), H3, C3, H4, C4, Probs4), generate_part(story_spine(5), H4, C4, H5, C5, Probs5), generate_part(story_spine(6), H5, C5, H6, C6, Probs6), {append([Probs1, Probs2, Probs3, Probs4, Probs5, Probs6], Probs)}.


generate_part(story_spine(1), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [there, once, was], 0, Hout, Cout, Probs).
generate_part(story_spine(2), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [every, day], 0.2, Hout, Cout, Probs).
generate_part(story_spine(3), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [but, one, day], -0.8, Hout, Cout, Probs).

% Part 4 can be repeated
generate_part(story_spine(4), Hin, Cin, Hout, Cout, Probs) --> {generate_condition(Random)}, lstm(Hin, Cin, [because, of, that], Random, Hint, Cint, Probs1), {random(R)}, test_for_repetition(story_spine(4), R, Hint, Cint, Hout, Cout, Probs2), {append([Probs1, Probs2], Probs)}.

generate_part(story_spine(5), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [finally], 0.8, Hout, Cout, Probs).
generate_part(story_spine(6), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [ever, since, then], 0.8, Hout, Cout, Probs).


test_for_repetition(story_spine(4), R, Hout, Cout, Hout, Cout, []) --> {R =< 0.5}, [].
test_for_repetition(story_spine(4), R, Hint, Cint, Hout, Cout, Probs) --> {R > 0.5}, generate_part(story_spine(4), Hint, Cint, Hout, Cout, Probs).


lstm(Hin, Cin, Xin, Cond, Hout, Cout, Probs) --> {py_call(use_sentiment_writingprompts_grammar:generate(Hin, Cin, Xin, Cond), -(Hout, Cout, Generated, Probs))}, Generated.


% useful predicates
generate_condition(Random) :- 
    random(R), 
    Random is 2 * (R - 1/2).