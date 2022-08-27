# Exemplo: manipulando vídeos com OpenCV (versão Python).

import cv2
import numpy as np

ESC_KEY = 27 # Código da tecla ESC.

# O vídeo de entrada pode ser um número ou um nome de arquivo. Se for um número,
# o programa tentará abrir a câmera com o número dado.
IN_VIDEO = 0

# Abrindo o vídeo. Ele volta como um objeto do tipo VideoCapture.
in_video = cv2.VideoCapture (IN_VIDEO)

# Se o vídeo vier da webcam, podemos tentar definir a resolução (depende da câmera).
if type (IN_VIDEO) is int:
    in_video.set (cv2.CAP_PROP_FRAME_WIDTH, 1280);
    in_video.set (cv2.CAP_PROP_FRAME_HEIGHT, 720);

# Pega a resolução do video.
rows = int (in_video.get (cv2.CAP_PROP_FRAME_HEIGHT))
cols = int (in_video.get (cv2.CAP_PROP_FRAME_WIDTH))

# O FourCC é um código de 4 caracteres que identifica um codec de vídeo. Os
# formatos válidos dependem dos codecs que estão instalados no computador!
fourcc = cv2.VideoWriter_fourcc (*'XVID')

# Vamos salvar uma cópia do video.
out_video = cv2.VideoWriter ('out.avi', fourcc, 30, (cols, rows))

# Um buffer qualquer.
img_gs = np.empty ((rows, cols, 1), np.uint8)

# Vai abrindo frames enquanto estiverem disponíveis E enquanto a tecla ESC não
# for pressionada.
key = 0
ok, img = in_video.read ()
while ok and key != ESC_KEY:
    cv2.cvtColor (img, cv2.COLOR_BGR2GRAY, dst = img_gs)    
    cv2.imshow ('cor', img)
    cv2.imshow ('cinza', img_gs)
    
    out_video.write (img) # Salva o frame.
    ok, img = in_video.read () # Lê o próximo frame.
    key = cv2.waitKey (1000//30) # Para manter ~30 fps.

in_video.release () # Fecha os vídeos.
out_video.release ()
cv2.destroyAllWindows ()
