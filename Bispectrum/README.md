# TODOS:
- Adapter le makefile pour l'ordi de Pierre et sont ordinateur d'acquisition
- Vérifier si on peut faire mieux de découper les datas en n_chunks de longueur l_fft

# Commentaire pour Pierre
- Doit être compiler sur l'ordinateur d'aquistion
- La première exécution est lente => FFTW calcul la méthode optimal pour faire les FFWT et enregistre un fichier FFTW_Wisdom.dat
	- Attention, le wisdom ne peut pas etre partager entre pc.
- L'execution de la fonction pourrait encore être accéléré
	- Ceci est important si tu planifi appeler souvent la fonction
	- On peut éviter de faire de l'allocation de memoire à chaque fois pour les variables intermédiares
		- Faire ceci nous permettrais aussi facilement d'accumuler l'histogram de nombre complexe en C++
			et le retourner à Python seulement une fois que le l'expérience est finit.
	- Le plan pourrait n'etre créé qu'une seule fois (et non a chaque fois que la fonction est appelé)
	- On pourrait utilser des commandes SIMD pour les produits somme etc. (je sais pas encore faire)
	
