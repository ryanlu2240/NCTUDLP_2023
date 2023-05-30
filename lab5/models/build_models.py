## Self-defined
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_encoder, vgg_decoder

def init_weights(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 or classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
		
def build_models(args, saved_model, device):
	print("\nBuilding models...")

	######################
	## Build the models ##
	######################
	if args.model_dir != "":
		frame_predictor = saved_model["frame_predictor"]
		posterior = saved_model["posterior"]
	else:
		frame_predictor = lstm(input_size=args.g_dim + args.z_dim + args.cond_dim, output_size=args.g_dim, hidden_size=args.rnn_size, 
			 					n_layers=args.predictor_rnn_layers, batch_size=args.batch_size, device=device)
		posterior = gaussian_lstm(input_size=args.g_dim, output_size=args.z_dim, hidden_size=args.rnn_size, 
			    				n_layers=args.posterior_rnn_layers, batch_size=args.batch_size, device=device)
		frame_predictor.apply(init_weights)
		posterior.apply(init_weights)
			
	if args.model_dir != "":
		decoder = saved_model["decoder"]
		encoder = saved_model["encoder"]
	else:
		encoder = vgg_encoder(args.g_dim)
		decoder = vgg_decoder(args.g_dim)
		encoder.apply(init_weights)
		decoder.apply(init_weights)
	
	########################
	## Transfer to device ##
	########################
	frame_predictor.to(device)
	posterior.to(device)
	encoder.to(device)
	decoder.to(device)

	return frame_predictor, posterior, encoder, decoder