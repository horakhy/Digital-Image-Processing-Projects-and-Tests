// Exemplo: manipulando vídeos com OpenCV (versão C++).

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp" // O módulo imgproc contém funções gerais para processamento de imagens.
#include "opencv2/highgui/highgui.hpp" // O módulo highgui contém funções básicas para I/O e apresentação de resultados.

#define ESC_KEY 27 // Código da tecla ESC.
#define IN_VIDEO 0 // O vídeo de entrada pode ser um número ou um nome de arquivo. Se for um número, o programa tentará abrir a câmera com o número dado.

using namespace std;
using namespace cv; // Namespace do OpenCV.

int main ()
{
	VideoCapture in_video (IN_VIDEO);

	// Se o vídeo vier da webcam, podemos tentar definir a resolução (depende da câmera).
	if (IN_VIDEO == 0)
	{
		in_video.set (CAP_PROP_FRAME_WIDTH, 1280);
		in_video.set (CAP_PROP_FRAME_HEIGHT, 720);
	}

	if (!in_video.isOpened ())
	{
		cerr << "Nao conseguiu usar a webcam." << endl;
		return (1);
	}

	// Pega a resolução do video.
	int rows = int (in_video.get (CAP_PROP_FRAME_HEIGHT));
	int cols = int (in_video.get (CAP_PROP_FRAME_WIDTH));

	// O FourCC é um código de 4 caracteres que identifica um codec de vídeo.
	// Os formatos válidos dependem dos codecs que estão instalados no computador!
	int fourcc = VideoWriter::fourcc ('x','v','i','d');

	// Vamos salvar uma cópia do video.
	VideoWriter out_video ("out.avi", fourcc, 30, Size (cols, rows), 1);
	if (!out_video.isOpened ())
	{
		cerr << "Nao conseguiu abrir o arquivo para escrita!\n";
		return (1);
	}

	// Um buffer qualquer.
	Mat img_gs (rows, cols, CV_8UC1);

	// Vai abrindo frames enquanto estiverem disponíveis E enquanto a tecla ESC não for pressionada.
	char key = 0;
	Mat img;
	in_video.read (img);
	while (img.data && key != ESC_KEY)
	{
		cvtColor (img, img_gs, COLOR_BGR2GRAY);

		imshow ("cor", img);
		imshow ("cinza", img);

		out_video.write (img); // Salva o frame.
		in_video.read (img); // Lê o próximo frame.

		key = waitKey (1000/30); // Para manter ~30 fps.
	}
	
	in_video.release (); // Fecha os vídeos.
	out_video.release ();
	destroyAllWindows ();
	return (0);
}
