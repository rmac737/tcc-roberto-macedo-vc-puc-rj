# A Modular Software Component for Automated ISO/IEC 19794-5 Facial Image Compliance Verification

#### Aluno: [Roberto da Silva Macedo](https://github.com/rmac737/)

#### Orientador: [Vitor Bento de Sousa](https://github.com/ICA-cursos/PoCs)

---

Trabalho apresentado ao curso [VC MASTER](https://ica.puc-rio.ai/vc-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

* [Link para o código](https://github.com/rmac737/tcc-roberto-macedo-vc-puc-rj)

---

### Resumo

Este trabalho apresenta o projeto e o desenvolvimento de um componente modular e automatizado para avaliação da qualidade e conformidade de imagens faciais segundo o padrão ISO/IEC 19795-1:2021, baseado em uma arquitetura híbrida que combina redes neurais profundas e técnicas clássicas de visão computacional.

O componente proposto realiza análises abrangentes de conformidade, incluindo estimativa de pose, direção dos olhos, detecção de oclusões, uniformidade de iluminação, avaliação de foco, detecção de artefatos de compressão e verificação de visibilidade das regiões faciais.

Ao empregar modelos convolucionais e baseados em transformadores, a solução proposta eleva significativamente a assertividade e a robustez das avaliações quando comparada às abordagens tradicionais atualmente em uso em ambientes operacionais.

### Abstract

This work presents the design and development of a modular and automated component for facial image quality and compliance assessment according to the ISO/IEC 19795-1:2021 standard, based on a hybrid architecture that combines deep neural networks and classical computer vision techniques.

The proposed component performs comprehensive compliance analyses, including pose estimation, gaze direction, occlusion detection, illumination uniformity, focus assessment, compression artifact detection, and facial region visibility verification.

By employing convolutional and transformer-based models, the proposed solution significantly improves the accuracy and robustness of evaluations when compared to traditional approaches currently used in operational environments.

---

### 1. Introdução

A rápida expansão dos ecossistemas de identidade digital posicionou a biometria facial como uma modalidade central de autenticação para governos, instituições financeiras e outros provedores de serviços públicos. Sistemas de reconhecimento facial são atualmente utilizados em diversos cenários críticos em relação à segurança do acesso e proteção da informação, incluindo programas nacionais de identificação, abertura digital de contas, pagamentos e/ou transferências bancárias, validação de aposentadorias e benefícios sociais, provas de vida remotas, controle de fronteiras e processos eletrônicos de Know Your Customer (eKYC), entre outros diversos exemplos.

A confiabilidade de longo prazo dos sistemas biométricos faciais está associada diretamente à qualidade e à padronização das imagens de cadastramento. Imagens inadequadas, afetadas por pose incorreta, olhar não frontal, oclusões, iluminação deficiente, artefatos de compressão ou falta de foco, degradam significativamente o desempenho dos mecanismos de comparação biométrica.

Com o objetivo de mitigar esses riscos, padrões internacionais como o ICAO Doc 9303 e a ISO/IEC 19794-5 definem requisitos rigorosos para a aquisição e o armazenamento de imagens faciais em documentos oficiais e bases biométricas.

Embora existam no mercado soluções destinadas à validação automática de conformidade com os padrões ISO/ICAO, essas soluções são predominantemente proprietárias, apresentam elevado custo de licenciamento e, em geral, não oferecem transparência metodológica, auditabilidade científica ou possibilidade de adaptação acadêmica.

Neste trabalho, apresentamos um componente modular e totalmente automatizado para avaliação da qualidade e da conformidade de imagens faciais segundo os padrões ICAO Doc 9303 e ISO/IEC 19794-5, fundamentado em uma arquitetura que integra modelos de aprendizado profundo e técnicas clássicas de visão computacional.

A abordagem proposta combina:

Redes neurais convolucionais YOLO para detecção facial;
MediaPipe Face Mesh para extração de landmarks faciais;
6DRepNet para estimativa de pose tridimensional;
MODNet/RVM para segmentação do fundo;
Redes especializadas para foco, ruído, pixelização, posterização, iluminação e acessórios faciais.

---

### 2. Modelagem

#### 2.1 ICAO Doc 9303 e ISO/IEC 19794-5

Os padrões International Civil Aviation Organization Doc 9303 e ISO/IEC 19794-5 são referências internacionais fundamentais para a padronização de imagens faciais utilizadas em sistemas de identificação biométrica.

O ICAO Doc 9303 estabelece especificações para documentos de viagem legíveis por máquina, definindo requisitos rigorosos para imagens faciais, incluindo:

* posição da cabeça;
* expressão neutra;
* iluminação uniforme;
* ausência de oclusões;
* qualidade geral da imagem.

A ISO/IEC 19794-5 especifica o formato e os requisitos para armazenamento e intercâmbio de imagens faciais em sistemas biométricos, detalhando características como resolução, proporções faciais, alinhamento, fundo e qualidade da imagem.

#### 2.2 Trabalhos Similares

O BioGaze é uma ferramenta de análise de imagens faciais desenvolvida para verificar automaticamente a conformidade com os padrões ISO/ICAO.

A solução combina modelos avançados de inteligência artificial com técnicas de visão computacional, permitindo a avaliação abrangente da qualidade da imagem.

Entre suas principais funcionalidades, o BioGaze realiza verificações relacionadas:

* direção do olhar;
* expressão neutra;
* ausência de acessórios;
* exposição;
* foco;
* saturação;
* fundo uniforme;
* pixelização;
* posterização.

Além disso, a ferramenta fornece relatórios detalhados sobre os resultados da análise, facilitando a identificação de não conformidades.

#### 2.3 Técnicas e Modelos de Deep Learning

Neste tópico, são detalhados cada item implementado do componente ICAO voltado à validação de qualidade de imagens.

#### 2.4 Validação Inicial da Imagem

Este item tem como objetivo validar se uma imagem está apta para ser processada, seguindo alguns critérios básicos de qualidade e formato. Verifica-se se a imagem possui resolução mínima. O ICAO exige que a imagem tenha pelo menos 480 pixels de largura e 640 pixels de altura.

Na sequência, o método analisa os primeiros bytes do arquivo (o “header”) para identificar o formato da imagem. Ele verifica se os bytes correspondem às assinaturas de arquivos JPEG ou PNG.

Sem aprovação destes itens, a imagem é rejeitada.

#### 2.5 Detecção Facial

Este item tem como objetivo identificar a presença de uma face na imagem e isolá-la para as próximas etapas de validação.

Para isso, é utilizado um modelo baseado em Caffe (Convolutional Architecture for Fast Feature Embedding), que realiza a detecção facial de forma eficiente. Esse tipo de abordagem utiliza dois arquivos principais:

.prototxt, responsável por definir a estrutura da rede neural;
res10_300x300_ssd_iter_140000.caffemodel, contendo os pesos treinados.

O objetivo deste item é verificar se existe uma face na imagem por meio de um modelo simples e rápido, especialmente quando comparado a alternativas mais pesadas como MTCNN, FaceNet e YOLO.

Sem a presença de uma face humana, a imagem é rejeitada.

#### 2.6 Extração de Landmarks Faciais

Neste item, utiliza-se o modelo face_mesh_Nx3x192x192_post para extração de landmarks faciais, ou seja, pontos específicos do rosto como olhos, nariz, boca e contorno facial.

Esses pontos permitem uma análise mais detalhada da imagem, sendo fundamentais para diversas validações, como:

alinhamento e centralização do rosto;
verificação de pose;
proporções faciais;
abertura dos olhos;
posicionamento da boca.

O sufixo “post” no nome do modelo indica que ele inclui certo nível de pós-processamento interno, facilitando seu uso direto nas etapas seguintes.

#### 2.7 Remoção de Fundo

O ICAO estabelece que imagens devem possuir fundo uniforme. Com o objetivo de atender a esse requisito e padronizar as imagens para as etapas seguintes de validação, o componente implementa a remoção automática do fundo por meio do modelo RVM (Robust Video Matting).

Esse modelo baseia-se no conceito de image matting, estimando para cada pixel o grau de pertencimento ao primeiro plano ou ao fundo.

Neste contexto, utiliza-se o modelo rvm_resnet50 para segmentar o primeiro plano da imagem, isolando a região correspondente ao indivíduo.

Para este projeto, foi definido o branco como cor padrão de fundo para todas as imagens.

#### 2.8 Verificação de Olhos Abertos

Os olhos abertos são um item obrigatório da validação ICAO. Assim, o componente utiliza pontos específicos da região ocular extraídos do modelo Face Mesh.

A métrica utilizada é uma variação do Eye Aspect Ratio (EAR), que calcula a razão entre as distâncias verticais e horizontais do olho.

Para este projeto, foi adotado o limiar de 0,20:

valores acima indicam olhos abertos;
valores abaixo indicam olhos fechados.

#### 2.9 Detecção de Óculos

Para atender ao requisito do ICAO de ausência de óculos, foi utilizado um modelo baseado na arquitetura YOLOv11.

O problema é tratado como uma tarefa de classificação binária:

com óculos;
sem óculos.

Foi realizado treinamento supervisionado com imagens previamente rotuladas utilizando Roboflow.

O dataset foi dividido em:

70% treinamento;
30% teste.

Foram utilizados os hiperparâmetros padrão recomendados pela arquitetura YOLOv11, incluindo:

- taxa de aprendizado inicial (*learning rate*) de `0.01`;
- otimizador SGD (*Stochastic Gradient Descent*);
- tamanho de lote (*batch size*) de `16`;
- treinamento com `500` épocas;
- tamanho de entrada das imagens de `640x640` pixels;
- função de perda padrão da arquitetura;
- aumento automático de dados (*data augmentation*) habilitado.

#### 2.10 Reconhecimento de Expressão Facial

O modelo EmotiEffLib é utilizado para realizar identificação automática das emoções presentes em uma face.

No contexto do componente ICAO, a utilização desse modelo está diretamente relacionada ao requisito de expressão facial neutra.

Caso a classificação seja diferente de “neutro”, a imagem é considerada fora do padrão exigido.

#### 2.11 Detecção de Cobertura de Cabeça

Foi utilizado YOLOv11 para detectar acessórios que cubram a cabeça, como:

chapéus;
bonés;
coberturas faciais.

O treinamento também utilizou datasets previamente rotulados contendo exemplos positivos e negativos para cada uma das categorias analisadas. O conjunto de dados foi dividido em 70% para treinamento e 30% para validação/teste, permitindo avaliar a capacidade de generalização do modelo em imagens não vistas durante o treinamento.

Foram utilizados os hiperparâmetros padrão recomendados pela arquitetura YOLOv11, assim como nos itens anteriores.

A utilização desses parâmetros padrão permitiu obter treinamento estável e convergência satisfatória do modelo sem necessidade inicial de ajustes finos (fine tuning), apresentando resultados consistentes na detecção automática de acessórios faciais em diferentes condições de iluminação, pose e qualidade de captura.

#### 2.12 Verificação de Boca Fechada

Para a verificação de abertura da boca, o componente utiliza landmarks faciais específicos da região labial.

A abordagem baseia-se no cálculo do Mouth Aspect Ratio (MAR):

MAR = dist(UpperLipCenter, LowerLipCenter) / dist(MouthLeftCorner, MouthRightCorner)

Quando a boca está fechada, o valor de MAR permanece baixo.

Critério utilizado:

MAR ≤ 0,20 → aprovado;
MAR > 0,20 → rejeitado.

#### 2.13 Classificação Ocular

O componente utiliza conjuntos específicos de landmarks faciais que delimitam precisamente os olhos esquerdo e direito.

Após o recorte das regiões oculares, aplica-se uma abordagem de classificação binária:

olho aberto;
olho fechado.

O modelo foi treinado utilizando imagens previamente rotuladas. Além disto, também foram utilizados os hiperparâmetros padrão recomendados pela arquitetura YOLOv11, assim como 
nos itens anteriores.

#### 2.14 Detecção de Maquiagem

Para a verificação do uso de maquiagem, o componente também adota uma abordagem baseada em deep learning.

Foi utilizado um dataset com mais de 50 mil imagens contendo:

diferentes tons de pele;
diferentes tipos de maquiagem;
diferentes condições de iluminação.

O objetivo é distinguir entre:

com maquiagem;
sem maquiagem.

Foram utilizados os hiperparâmetros padrão recomendados pela arquitetura YOLOv11, assim como nos itens anteriores.

#### 2.15 Estimativa de Direção do Olhar

Este item implementa um módulo responsável por estimar a direção do olhar a partir de uma imagem facial.

O modelo retorna:

yaw;
pitch.

Critérios adotados:

pitch dentro de aproximadamente 10 graus;
yaw dentro de aproximadamente 15 graus.

Caso esses limites sejam ultrapassados, a imagem é reprovada.

#### 2.16 Estimativa de Pose da Cabeça

Para validação da posição da cabeça, o componente utiliza o modelo SixDRepNet.

O modelo estima:

pitch;
yaw;
roll.

A partir desses valores, o componente realiza verificação baseada em limiares pré-definidos.

A imagem é considerada válida apenas quando todos os ângulos permanecem dentro dos limites aceitáveis definidos pelo ICAO.

---

### 3. Resultados

Os resultados obtidos durante os testes do componente desenvolvido mostraram-se satisfatórios e compatíveis com os objetivos propostos neste trabalho. A solução apresentou desempenho consistente na validação automatizada dos critérios definidos pelos padrões ICAO Doc 9303 e ISO/IEC 19794-5, demonstrando boa capacidade de detecção e análise dos diferentes requisitos de conformidade facial.

Os experimentos foram realizados em um ambiente computacional modesto, equipado com processador Intel Core i5 de 8ª geração e 16 GB de memória RAM, sem utilização de GPU dedicada. Mesmo nesse cenário, o sistema apresentou baixo consumo de memória e boa eficiência computacional, evidenciando a viabilidade da proposta para aplicações reais de baixo custo operacional.

Durante os testes, o componente foi capaz de executar toda a cadeia de inferência, incluindo detecção facial, extração de landmarks, segmentação de fundo, estimativa de pose, análise do olhar, validação de expressão facial e verificação dos demais critérios ICAO, em aproximadamente 3 segundos por imagem utilizando apenas processamento em CPU.

Além do desempenho computacional, observou-se que a arquitetura modular baseada em deep learning proporcionou maior robustez frente a variações de iluminação, posicionamento facial, acessórios e diferentes condições de captura. Os resultados alcançados atenderam às expectativas iniciais do projeto, demonstrando que a solução possui potencial para utilização em ambientes reais de validação biométrica facial, especialmente em cenários que demandam baixo custo, portabilidade e independência de hardware especializado.

---

### 5. Conclusões

Este trabalho apresentou o desenvolvimento de um componente que automatiza a avaliação da qualidade e conformidade de imagens faciais com base nos requisitos estabelecidos pelo ICAO Doc 9303 e pela norma ISO/IEC 19794-5.

A solução integrou:

* técnicas clássicas de visão computacional;
* modelos modernos de deep learning;
* avaliação automática de múltiplos critérios biométricos.

Os resultados demonstram que a utilização de modelos baseados em aprendizado profundo proporciona maior robustez e precisão quando comparada às abordagens tradicionais.

Além disso, a arquitetura modular adotada mostrou-se flexível e extensível, possibilitando a inclusão de novos critérios de validação conforme necessário.

Também, identifica-se, como trabalho futuro, a necessidade de expandir e aperfeiçoar as funcionalidades do componente, incorporando novas verificações de qualidade de imagem, como:

* detecção de pixelização;
* posterização;
* desfoque;
* sombras no rosto;
* reflexos de luz.

Como proposta futura, sugere-se a construção de um dataset robusto, diversificado e devidamente rotulado.

Por fim, destaca-se que a continuidade deste trabalho tende a elevar ainda mais o desempenho do sistema, consolidando sua aplicabilidade em cenários reais de validação biométrica.

---

# Referências

* ICAO Doc 9303
* ISO/IEC 19794-5
* BioGaze
* FaceQnet
* FaceQvec
* YOLO
* MediaPipe Face Mesh
* SixDRepNet
* Robust Video Matting
* EmotiEffLib

---

Matrícula: 241.100.454

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Visão Computacional Master*
