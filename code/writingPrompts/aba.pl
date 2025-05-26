% Probs is just for calculating the perplexity

start_aba(HA0, CA0, HB0, CB0, HA2, CA2, HB1, CB1, Probs) --> generate_part(deel(1), HA0, CA0, HA1, CA1, Probs1), generate_part(deel(2), HB0, CB0, HB1, CB1, Probs2), generate_part(deel(3), HA1, CA1, HA2, CA2, Probs3), {append([Probs1, Probs2, Probs3], Probs)}.


generate_part(deel(1), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [first], 0.5, Hout, Cout, Probs).
generate_part(deel(2), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [next], -0.5, Hout, Cout, Probs).
generate_part(deel(3), Hin, Cin, Hout, Cout, Probs) --> lstm(Hin, Cin, [finally], 0.5, Hout, Cout, Probs).


lstm(Hin, Cin, Xin, Cond, Hout, Cout, Probs) --> {py_call(use_sentiment_writingprompts_grammar:generate(Hin, Cin, Xin, Cond), -(Hout, Cout, Generated, Probs))}, Generated.
