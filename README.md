# PlacaDetect

Sistema para automatizar a detecção e ofuscação (blur/pixelização) de placas de
motos/carros em grandes volumes de fotos — pensado para o fluxo de um
fotógrafo que precisa preparar milhares de fotos para venda no site sem
expor a placa dos veículos.

Você aponta uma pasta com as fotos (ou faz upload), clica em **Processar** e
o sistema:

1. Detecta automaticamente a placa em cada foto usando um modelo de IA
   (YOLOv9 especializado em placas veiculares).
2. Aplica desfoque (ou pixelização, ou uma caixa preta) sobre a placa.
3. Salva a foto processada, pronta para upload no seu site.
4. Mostra no painel o resultado de cada foto: sucesso, nenhuma placa
   encontrada, ou erro — para você revisar antes de publicar.

Todo o processamento roda localmente no seu computador (nenhuma foto é
enviada para serviços externos).

## Como funciona (visão geral)

```
Fotos (pasta local ou upload)
        │
        ▼
  Detector de placas (IA, YOLOv9 – modelo "yolo-v9-t-640-license-plate-end2end")
        │
        ├─ placa encontrada  → aplica blur/pixelização na região da placa → data/output/
        └─ nenhuma placa     → copia a foto original (redimensionada) → data/output/
        │
        ▼
  Painel web: status de cada foto + download em lote (.zip)
```

- **Backend**: Python + FastAPI. Detecção com [`open-image-models`](https://github.com/ankandrew/open-image-models)
  (YOLOv9 treinado especificamente para placas, roda em CPU via ONNX Runtime —
  não precisa de GPU). O modelo (~7 MB) é baixado automaticamente na primeira
  execução e fica em cache em `~/.cache/open-image-models`.
- **Frontend**: HTML/CSS/JS puro (sem build step), servido pelo próprio backend.
- **Banco de dados**: SQLite local (`data/placadetect.db`) guardando o status
  de cada foto processada.

## Instalação

Requer Python 3.10+.

```bash
cd PlacaDetect
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Uso

```bash
python3 run.py
```

Isso sobe o servidor em `http://localhost:8000` e abre o navegador
automaticamente. Na primeira execução, o download do modelo de IA (~7 MB)
pode levar alguns segundos.

Na interface:

1. **Importar fotos**
   - **Pasta local**: cole o caminho da pasta onde estão as 5 mil fotos
     (ex.: `/home/usuario/fotos-evento` ou `C:\Fotos\evento`) e clique em
     "Escanear pasta". As fotos originais **não são movidas nem alteradas**
     — o sistema só lê os arquivos e escreve as versões processadas em
     `data/output/`.
   - **Upload**: arraste fotos direto no navegador (bom para testes ou lotes
     pequenos).
2. **Processar**: escolha o estilo de redação (desfoque, pixelização ou caixa
   preta) e clique em "Processar pendentes". Uma barra de progresso mostra o
   andamento em tempo real; o processamento roda em segundo plano e pode ser
   cancelado a qualquer momento.
3. **Resultados**: cada foto aparece com um selo de status:
   - ✅ **Sucesso** — placa encontrada e ofuscada.
   - ⚠️ **Sem placa** — nenhuma placa foi detectada (vale revisar manualmente
     essas fotos antes de publicar).
   - ❌ **Erro** — falha ao processar (ex.: arquivo corrompido); pode ser
     reprocessada com "Reprocessar erros".

   Clique em qualquer foto para ver o comparativo antes/depois e baixar em
   resolução original. Use "Baixar prontos (.zip)" para baixar todas as fotos
   com status de sucesso de uma vez, prontas para subir no site.

## Ajustando a sensibilidade / qualidade da detecção

Na seção "Processar" da interface:

- **Sensibilidade de detecção**: controla o quão "exigente" o modelo é.
  "Alta" detecta mais placas, mas pode gerar mais falsos positivos; "Conservadora"
  é mais rígida. Se você notar muitas fotos caindo em "Sem placa" mas com
  placa visível, aumente a sensibilidade.
- **Limitação conhecida**: em raras fotos, uma placa perfeitamente legível
  não é detectada mesmo em qualquer sensibilidade — não é um problema de
  limiar/configuração, é o modelo de detecção em si não reconhecendo aquela
  placa específica. Testado a fundo num caso real (placa numa moto com
  suporte que a projeta para longe do rabeta, ao lado de um carro claro ao
  fundo): nem isolar a placa num recorte bem ampliado, nem trocar para
  outro modelo/resolução disponível na biblioteca resolveu de forma
  confiável — um modelo alternativo até encontra a placa às vezes, mas
  junto traz vários falsos positivos novos em qualquer foto, então trocar
  não compensa. Decisão consciente: manter o fluxo 100% automático mesmo
  assim (em vez de adicionar ajuste manual na interface) — quando acontecer,
  revise manualmente o resultado antes de publicar a foto, já que o
  "Sucesso"/"Sem placa" reflete só as placas que o modelo conseguiu
  identificar.
- **Redação da placa**: `blur` (desfoque forte, padrão), `pixelate`
  (mosaico) ou `black` (caixa sólida). Todas garantem que a placa fique
  ilegível; a região tratada é exatamente o contorno real da placa (ver
  abaixo) — nada de margem pra fora dele. A borda é suavizada só PRA DENTRO
  desse contorno (uma faixa fina próxima à borda fica com opacidade um
  pouco menor, em vez de um corte seco de 100%→0%), então a transição funde
  com a própria placa sem nunca vazar desfoque sobre o que está ao redor
  dela (guidão, pneu, parede).
- **Cabeçalho "BRASIL" preservado**: em placas padrão Mercosul, a faixa azul
  no topo (BRASIL + QR) é identificada por cor e mantida visível — só a
  parte com letras/números é redigida, no mesmo espírito de como as placas
  antigas mantinham a cidade-UF visível. Isso só acontece quando a faixa
  azul é identificada com confiança (mesmo critério de "seguro por padrão"
  do resto do sistema); sem essa confiança — placa suja, embaçada, padrão
  antigo sem essa faixa — a placa inteira é redigida, sem tentar adivinhar
  onde cortar.

Melhorias adicionais que rodam automaticamente, sem configuração:

- **Fotos com várias motos no quadro**: quando a imagem é bem maior que a
  resolução nativa do modelo (comum em fotos horizontais com várias motos
  lado a lado), o sistema também roda a detecção em recortes sobrepostos da
  imagem, no tamanho nativo do modelo, além da imagem inteira — placas
  distantes que encolheriam demais numa única passada continuam
  detectáveis, e como esses recortes já veem a placa em resolução mais alta,
  a caixa que eles devolvem é usada no lugar da caixa (menos precisa) da
  passada de imagem inteira sempre que as duas se sobrepõem — evita o
  desfoque cair ao lado da placa em vez de em cima dela.
- **Menos falsos positivos, sem descartar placas de moto reais**: detecções
  muito alongadas (faixa/friso comprido na moto) são sempre descartadas.
  Formas quase quadradas — o caso de um adesivo/refletor redondo, mas também
  o caso normal de uma placa de moto (já quase quadrada) fotografada em
  ângulo — só são descartadas quando a confiança do modelo é baixa; com
  confiança razoável, a forma quase quadrada é aceita como placa.
- **Sem detecção duplicada na mesma placa**: quando a passada de imagem
  inteira e um recorte detectam a mesma placa física com caixas que não se
  sobrepõem o bastante para o algoritmo de deduplicação padrão (comum
  quando a caixa da imagem inteira é imprecisa), as duas eram tratadas como
  placas diferentes e as DUAS eram borradas — resultando numa área de
  desfoque maior e mais deslocada do que a placa real, além do contador de
  detecções ficar errado. O critério de sobreposição agora considera o
  quanto do menor dos dois boxes está coberto, não a interseção sobre a
  união (IoU) — resolve isso sem exigir que os dois boxes tenham tamanho
  parecido.
- **Desfoque acompanha a inclinação real da placa**: em vez de sempre borrar
  a caixa alinhada aos eixos do detector (que numa moto em ângulo sobra
  fundo nos quatro cantos), o sistema tenta reconstruir o retângulo real
  (girado) da placa. Duas estratégias, na ordem:
  1. **Via faixa azul Mercosul**: a faixa "BRASIL" é um alvo de cor bem mais
     fácil de isolar do que a placa inteira (não se mistura com pneu/
     carenagem/asfalto do jeito que o contorno da placa toda se mistura).
     A inclinação e a largura exatas da faixa, combinadas com a área da
     caixa do detector, bastam pra reconstruir o retângulo completo da
     placa — funciona bem mesmo com a moto bem inclinada.
  2. **Contorno genérico** (Canny + aproximação poligonal) como plano B,
     pra placas sem faixa azul identificável. Mais frágil — cai pra caixa
     alinhada aos eixos com frequência em fotos reais (barro, reflexo,
     objetos encostados na placa atrapalham o contorno).

  Sempre que nenhuma das duas encontra um resultado confiável, usa a caixa
  alinhada aos eixos do detector — mais fundo ao redor da placa do que o
  ideal, mas nunca deixa parte da placa de fora do desfoque. Em nenhum dos
  casos a área tratada ultrapassa o contorno usado: não existe mais uma
  margem extra "por segurança" que espalhe o desfoque para além da placa
  (isso chegou a acontecer e foi reportado — o desfoque vazava visivelmente
  sobre o pneu/carenagem ao redor; a margem de padding foi removida, a
  suavização da borda agora fica inteiramente contida dentro do contorno).

Variáveis de ambiente (opcionais, definidas antes de rodar `python3 run.py`):

| Variável | Padrão | Descrição |
|---|---|---|
| `PLACADETECT_MODEL` | `yolo-v9-t-640-license-plate-end2end` | Modelo de detecção. Modelos maiores (`yolo-v9-s-608-license-plate-end2end`) são mais precisos porém mais lentos. |
| `PLACADETECT_CONF_THRESH` | `0.3` | Confiança mínima para considerar uma detecção válida. |
| `PLACADETECT_REDACTION_STYLE` | `blur` | `blur`, `pixelate` ou `black`. |
| `PLACADETECT_BOX_PADDING` | `0.12` | Margem extra (%) ao redor da placa detectada — pequena de propósito, para a área tratada acompanhar o tamanho real da placa. |
| `PLACADETECT_DATA_DIR` | `./data` | Onde ficam o banco, as miniaturas e as fotos processadas. |

## Processando 5 mil fotos

O modelo padrão (`yolo-v9-t-640`) é leve o suficiente para rodar em CPU comum
a alguns quadros por segundo — um lote de 5.000 fotos tende a levar de
alguns minutos a cerca de meia hora em um notebook comum, dependendo do
hardware. O processamento:

- roda em segundo plano (você pode fechar a aba e voltar depois — o painel
  mostra o progresso ao reabrir);
- é **resumível**: se você fechar o servidor no meio do lote, as fotos ainda
  não processadas continuam marcadas como "pendente" e são retomadas ao
  clicar em "Processar pendentes" de novo;
- não duplica fotos já importadas (rodar "Escanear pasta" de novo na mesma
  pasta não recria registros existentes).

## Testes

```bash
pip install pytest httpx
python3 -m pytest tests/ -v
```

O teste sobe a API completa (upload → processamento → verificação do
resultado) usando uma imagem sintética, validando a integração de ponta a
ponta do pipeline.

## Estrutura do projeto

```
PlacaDetect/
├── backend/
│   ├── app/
│   │   ├── main.py        # rotas da API (FastAPI)
│   │   ├── config.py      # configurações (paths, modelo, thresholds)
│   │   ├── database.py    # acesso ao SQLite
│   │   ├── detector.py    # wrapper do modelo de detecção de placas
│   │   ├── imaging.py     # carregamento (com correção de EXIF), blur/pixelização, thumbnails
│   │   ├── processor.py   # orquestra o processamento em lote (thread de fundo + progresso)
│   │   └── ingest.py      # importação por pasta local / upload
│   └── requirements.txt
├── frontend/               # painel web (HTML/CSS/JS puro)
├── data/                   # banco + fotos processadas (gerado em runtime, fora do git)
├── tests/
└── run.py                  # ponto de entrada (sobe o servidor e abre o navegador)
```

## Limitações conhecidas / próximos passos

- O modelo detecta a **posição** da placa, mas não faz OCR — ele não sabe
  "ler" o texto, o que é bom para privacidade (não é necessário reconhecer o
  número da placa para apagá-la).
- Fotos com a placa muito pequena, muito desfocada por movimento, ou quase
  totalmente fora de quadro podem cair em "Sem placa" — por isso o painel
  separa esse status, para revisão manual rápida antes de publicar.
- Hoje o processamento é sequencial (uma foto por vez) para manter o código
  simples; se 5 mil fotos demorarem mais do que o desejável no seu hardware,
  dá para paralelizar o `processor.py` com um pool de workers.
