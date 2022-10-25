clc; 
clear all;


inputs=[0 0;0 1;1 0;1 1];
output=[0; 1; 1; 0];
weights_hidden=rand(2,2);
%weights_hidden=[0.5 0.5;0.5 0.5];
biases_hidden=rand(2,1);
%biases_hidden=[0.5; 0.5];
biases_output=rand(1,1);
%biases_output=0.5;
weights_output=rand(2,1);
%weights_output=[0.5; 0.5];


%sigmoid(weights_hidden.*inputs(1,:))

for j=1:10000

for i=1:4
[finaloutput,ActivationFunction1] = forwardprop(inputs(i,:),weights_hidden, biases_hidden, weights_output, biases_output);
[weights_outputnew,weights_hiddennew,biases_hiddennew,biases_outputnew]=backprop(output(i),finaloutput,ActivationFunction1,inputs(i,:),weights_hidden,weights_output,biases_hidden,biases_output,0.1);


weights_output=weights_outputnew;
weights_hidden=weights_hiddennew;
biases_output=biases_outputnew;
biases_hidden=biases_hiddennew;

inputs(i,:)
finaloutput
end

end


for i=1:4
[finaloutput,ActivationFunction1] = forwardprop(inputs(i,:),weights_hidden, biases_hidden, weights_output, biases_output);
finaloutput;
end



function [finaloutput,ActivationFunction1] = forwardprop(inputs,weights_hidden, biases_hidden, weights_output, biases_output)

sigmoid1 = @(x) 1./(1+exp(-x));
NetFunction1=weights_hidden*inputs(1,:)'+biases_hidden; %inputs are only mentioned here I believe
ActivationFunction1=sigmoid1(NetFunction1);

NetFunction2=ActivationFunction1'*weights_output+biases_output;
finaloutput=sigmoid1(NetFunction2);


end

function [weights_outputnew,weights_hiddennew,biases_hidden_new,biases_output_new] = backprop(output,finaloutput,ActivationFunction1,inputs,weights_hidden,weights_output,biases_hidden,biases_output,learningrate)

deltaw5=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*ActivationFunction1(1);
deltaw6=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*ActivationFunction1(2);

deltaw1=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*weights_output(1)*ActivationFunction1(1)*(1-ActivationFunction1(1))*inputs(1,1);
deltaw3=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*weights_output(1)*ActivationFunction1(1)*(1-ActivationFunction1(1))*inputs(1,2);


deltaw2=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*weights_output(2)*ActivationFunction1(2)*(1-ActivationFunction1(2))*inputs(1,1);
deltaw4=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*weights_output(2)*ActivationFunction1(2)*(1-ActivationFunction1(2))*inputs(1,2);

deltab3=(output(1)-finaloutput)*finaloutput*(1-finaloutput);
deltab2=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*weights_output(2)*ActivationFunction1(2)*(1-ActivationFunction1(2));
deltab1=(output(1)-finaloutput)*finaloutput*(1-finaloutput)*weights_output(1)*ActivationFunction1(1)*(1-ActivationFunction1(1));




deltaweightshidden=[deltaw1 deltaw3;deltaw2 deltaw4];
deltabiaseshidden=[deltab1;deltab2];
deltaweightsoutput=[deltaw5;deltaw6];
deltabiasesoutput=[deltab3];


weights_outputnew=weights_output+learningrate*deltaweightsoutput;
weights_hiddennew=weights_hidden+learningrate*deltaweightshidden;
biases_hidden_new=biases_hidden+learningrate*deltabiaseshidden;
biases_output_new=biases_output+learningrate*deltabiasesoutput;

end