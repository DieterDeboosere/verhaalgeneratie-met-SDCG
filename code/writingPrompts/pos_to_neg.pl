% Probs is just for calculating the perplexity

start_pos_to_neg(H0, C0, X0, H5, C5, X5, Probs) --> [X0], generate_part(deel(1), H0, C0, X0, H1, C1, X1, Probs1), generate_part(deel(2), H1, C1, X1, H2, C2, X2, Probs2), generate_part(deel(3), H2, C2, X2, H3, C3, X3, Probs3), generate_part(deel(4), H3, C3, X3, H4, C4, X4, Probs4), generate_part(deel(5), H4, C4, X4, H5, C5, X5, Probs5), {append([Probs1, Probs2, Probs3, Probs4, Probs5], Probs)}.


generate_part(deel(1), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 1, Hout, Cout, Xout, Probs).
generate_part(deel(2), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 0.5, Hout, Cout, Xout, Probs).
generate_part(deel(3), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, 0, Hout, Cout, Xout, Probs).
generate_part(deel(4), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, -0.5, Hout, Cout, Xout, Probs).
generate_part(deel(5), Hin, Cin, Xin, Hout, Cout, Xout, Probs) --> lstm(Hin, Cin, Xin, -1, Hout, Cout, Xout, Probs).


lstm(Hin, Cin, Xin, Cond, Hout, Cout, Xout, Probs) --> {py_call(use_sentiment_writingprompts_grammar_only_condition:generate(Hin, Cin, Xin, Cond), -(Hout, Cout, Xout, Generated, Probs))}, Generated.
