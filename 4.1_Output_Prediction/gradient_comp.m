function y=gradient_comp(alpha,beta,B,F,Z,p_squared_out,p_squared_in)
        H_now=B*F*((p_squared_out/p_squared_in)^(Z/2)-1);
        first_step_grad=2*alpha*H_now+beta;
        gradient_F=first_step_grad*B*((p_squared_out/p_squared_in)^(Z/2)-1);
        gradient_out=first_step_grad*B*F*Z/2/p_squared_out*((p_squared_out/p_squared_in)^(Z/2));
        gradient_in=-first_step_grad*B*F*Z/2/p_squared_in*((p_squared_out/p_squared_in)^(Z/2));
        y=[gradient_F,gradient_out,gradient_in];
end