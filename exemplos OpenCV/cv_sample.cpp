// Exemplo: manipulando imagens com OpenCV (versão C++).

#include <time.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp" // O módulo imgproc contém funções gerais para processamento de imagens.
#include "opencv2/highgui/highgui.hpp" // O módulo highgui contém funções básicas para I/O e apresentação de resultados.

#define IN_IMG "teste.jpg"

using namespace cv; // Namespace do OpenCV.
using namespace std;

int main ()
{
	// A classe Mat é usada para representar imagens e matrizes.
	// Ela tem um vetor de dados interno, que é desalocado automaticamente pelo destruidor da classe.
	// A função imread carrega uma imagem de um arquivo, alocando e preenchendo o vetor de dados.
	// Ela tem um segundo parâmetro opcional que indica o número de canais da imagem.
	// O valor default é CV_LOAD_IMAGE_COLOR, que lê as imagens com 3 canais (BGR).
	// Se o parâmetro for mudado para CV_LOAD_IMAGE_GRAYSCALE, a imagem é carregada em escala de cinza (1 canal).
	Mat img = imread (IN_IMG);
	if (!img.data) // Se falhar, img.data == NULL.
	{
		cerr << "Nao conseguiu ler " << IN_IMG << ".\n";
		return (1);
	}

	// Convertendo para float32. Dependendo da sequência de operações, e recomendável fazer esta conversão (mas nem sempre é necessario).
	img.convertTo (img, CV_32F, 1.0/255.0);

	// Para exemplificar a forma de acesso aos dados da imagem, vamos criar uma versão em escala de cinza dela. A conta é:
	//
	// cinza = 0.299r + 0.587g + 0.114b
	//
	// Lembrando que a ordem dos canais na imagem colorida é (B, G, R).
	// Faremos isso três vezes, para comparar o desempenho.
	Mat img_gs (img.rows, img.cols, CV_32FC1); // CV_32FC1 = float32, 1 canal.

	// Versão 1: usando a função at. Esta operação é lenta, e deve ser evitada quando possivel. Otimizações do compilador podem melhorar isso, mas não totalmente.
	clock_t start_time = clock ();
	for (int row = 0; row < img.rows; row++)
		for (int col = 0; col < img.cols; col++)
		{
			Vec3f pixel = img.at <Vec3f> (row, col); // Vec3f: 3-tupla de float32.
			img_gs.at <float> (row, col) = pixel [0]*0.114f + pixel [1]*0.587f + pixel [2]*0.299f;
		}
	cout << "Tempo com at: " << clock () - start_time << endl;

	// Versão 2: usando ponteiros. Quando for percorrer uma imagem pixel a pixel, faça sempre assim.
	start_time = clock ();
	int rows = img.rows;
	int cols = img.cols;
	if (img.isContinuous () && img_gs.isContinuous ()) // Truque: se o buffer for contínuo, coloca tudo em uma linha gigante.
	{
		cols *= rows;
		rows = 1;
	}

	for (int row = 0; row < rows; row++)
	{
		float* ptr_in = (float*) (img.data + row*img.step); // Posiciona o ponteiro no início da linha atual.
		float* ptr_out = (float*) (img_gs.data + row*img_gs.step);
		for (int col = 0; col < cols; col++)
		{
			*ptr_out = ptr_in [0]*0.114f + ptr_in [1]*0.587f + ptr_in [2]*0.299f;
			ptr_in += 3; // Pula 3 posições (3 canais).
			ptr_out++; // Pula 1 posição.
		}
	}
	cout << "Tempo com ponteiros: " << clock () - start_time << endl;

	// Versão 3: usando a função embutida do OpenCV.
	start_time = clock ();
	cvtColor (img, img_gs, COLOR_BGR2GRAY);
	cout << "Tempo OpenCV: " << clock () - start_time << endl;

	// Mostra os resultados.
	imshow ("cor", img);
	imshow ("cinza", img_gs);

	// Salva. Note que eu preciso multiplicar de novo por 255, mas não preciso reconverter para inteiro.
	imwrite ("cinza.png", img_gs * 255);

	// A função waitKey fica esperando o usuário pressionar uma tecla.
	// Ela recebe um parâmetro, com valor default 0, que diz quantos milissegundos a função espera - se o valor for menor ou igual a 0, a função espera indefinidamente.
	// O valor de retorno é o código ASCII da tecla pressionada - neste caso, não importa.
	// Experimente mudar o parâmetro para 1000 para ver o que acontece.
	waitKey ();

	destroyAllWindows ();
	return (0);
}