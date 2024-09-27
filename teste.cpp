/*
  Rede Neural Artificial Evolutiva (RNA-E)
  
  Os pesos são atualizados a partir de um algoritmo
  genético que busca minimizar os erros na fase de
  treinamento.
  
*/

//GABRIELLY ZENI MANTHAY

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_Entradas 2
#define MAX_Pesos 6

//===| Estrutura de Dados |=====================================================
typedef char string[60];

typedef struct tipoLicao {
	int p;  //proposição P
	int q;	//Proposição Q
	int resultadoEsperado; //Proposição Composta P "E" Q (A Classe)
	tipoLicao *prox;
}TLicao;

typedef struct tipoIndividuo {
	float genes[MAX_Pesos];
	int erros;
	int numero; //numero identificador
	tipoIndividuo  *prox;
}TIndividuo;

typedef struct tipoSinapse {
	int camada;
	int neuronio_origem;
	int neuronio_destino;
	float peso;
	tipoSinapse *prox;
}TSinapse;

typedef struct tipoNeuronio {
	int neuronio;
	float soma;
	float peso;
	tipoNeuronio *prox;
}TNeuronio;

typedef struct tipoLista{
	FILE *fp; //Arquivo de Saída (Relatório).
	string objetivo;
	TLicao *licoes; //Conjunto de lições a serem aprendidas
	float entradas[MAX_Entradas];
	TNeuronio *neuronios;
	TSinapse *pesos;
	TIndividuo *populacao;
	TIndividuo *individuoAtual;
	int totalIndividuos;
	int Qtd_Populacao;
	int Qtd_Mutacoes_por_vez;
	int Total_geracoes;
	int geracao_atual;
	int Qtd_Geracoes_para_Mutacoes; 
	float sinapseThreshold;
	float learningRate;
}TLista;

TLista lista;

//====| Assinatura de Funções |=================================================
void inicializa(TLista *L);
void geraIndividuos(TLista *L);
void geraLicoes(TLista *L);
void insereLicao(TLista *L, int p, int q, int resultado);
void insereNeuronio(TLista *L, int neuronio);
void estabelecendoSinapse(TLista *L,int neuronioDe, int neuronioAte, int camada);
void printIndividuos(TLista *L);
void treinamento(TLista *L);
void cruzamento(TLista *L);
void avaliacaoIndividuos(TLista *L);
void ordenamentoIndividuos(TLista *L);
void promoveMutacoes(TLista *L, float learningRate);
void poda(TLista *L);
//===| Programa Principal |=====================================================
int main(){
	inicializa(&lista);
	treinamento(&lista);
	
}
//===| Funções |================================================================
void inicializa(TLista *L){
	int i;
	
	L->licoes = NULL;
	L->populacao = NULL;
	L->individuoAtual = NULL;
	L->totalIndividuos = 0;
	
	for(i=0; i < MAX_Entradas; i++){
		L->entradas[i] = 0;
	}//for
	
	L->neuronios = NULL;
	L->pesos = NULL;
	
	printf("\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====");
	printf("\n\n\t\t=====| Configuracao da RNA |=====\n\n");
	printf("\tInforme o TAMANHO da POPULACAO (em termos de individuos):\n");
	printf("\t\tSugestao: 300 individuos.\n\t\tValor: ");
	scanf("%d", &L->Qtd_Populacao);
	
	geraIndividuos(L);
	
	printf("\n\n\tInforme a QUANTIDADE de GERACOES maxima:");
	printf("\n\tSugestao: 100 geracoes no total.\n\tValor: ");
	scanf("%d", &L->Total_geracoes);
	
	L->geracao_atual = 0;
	
	printf("\n\n\tInforme o INTERVALO de GERACOES para a ocorrencia de MUTACOES:");
	printf("\n\tSugestao: 5 (a cada 5 geracoes devem ocorrer mutacoes).\n\tValor: ");
	scanf("%d", &L->Qtd_Geracoes_para_Mutacoes);
	
	printf("\n\n\tInforme a QUANTIDADE de MUTACOES que devem ocorrer POR VEZ:");
	printf("\n\tSugestao: 3 mutacoes por intervalo.\n\tValor: ");
	scanf("%d", &L->Qtd_Mutacoes_por_vez);
	
	printf("\n\nSINAPSE THRESHOLD (Limiar das Conexoes entre Neuronios):\n");
	printf("Define a intensidade do sinal que sensibiliza cada neuronio.\n\n");
	printf("\tInforme o SINAPSE THRESHOLD:\n\tSugestao: 0.60\n\tValor: ");
	scanf("%f",&L->sinapseThreshold);
	
	printf("\n\nLEARNING RATE (Taxa de Aprendizado): variacao dos pesos em cada ajuste (Aprendizado).\n");
	printf("\n\tLEARNING RATE:\n\tSugestao: 0.20\n\tValor: ");
	scanf("%f",&L->learningRate);
	
	strcpy(L->objetivo,"Aprendizado da Funcao Logica P E Q");
	
	printf("\n\n\tDefinindo as LICOES a serem aprendidas pela Rede Neural Artificial.\n\n");
	geraLicoes(L);
	
	printf("\n\n\tDefinindo os NEURONIOS que compoem a REDE NEURAL ARTIFICIAL.");
	insereNeuronio(L, 1);
	insereNeuronio(L, 2);
	insereNeuronio(L, 3);
	insereNeuronio(L, 4);
	insereNeuronio(L, 5);
	
	printf("\n\n\tEstabelecendo as CONEXOES (Sinapses) entre os NEURONIOS.");
	estabelecendoSinapse(L,1,3,0);
	estabelecendoSinapse(L,1,4,0);
	estabelecendoSinapse(L,2,3,0);
	estabelecendoSinapse(L,2,4,0);
	estabelecendoSinapse(L,3,5,1);
	estabelecendoSinapse(L,4,5,1);
	
	L->fp = fopen("RNA_EVOLUTIVA_RELATORIO_Gabrielly.TXT","w");
	
	fprintf(L->fp,"\n\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====\n\n");
	fprintf(L->fp,"\tOBJETIVO: %s.\n\n\tLicoes:\n", L->objetivo);
	fprintf(L->fp,"\t LICAO    P    Q  (Resultado Esperado)\n");
	fprintf(L->fp,"\t+------+----+----+---------------------+\n");
	
	TLicao *licao = L->licoes;
	int cont = 0;
	while(licao != NULL){
		fprintf(L->fp,"\t(%d) - %d   %d   %d\n", ++cont, licao->p, licao->q, licao->resultadoEsperado);
		licao = licao->prox;
	}//while
	
	fprintf(L->fp,"\n\n");
	fprintf(L->fp,"\tLearning Rate: %.2f\n", L->learningRate);
	fprintf(L->fp,"\tSinapse Threshold: %.2f\n", L->sinapseThreshold);
	fprintf(L->fp,"\tPopulacao MAXIMA: %d.\n", L->Qtd_Populacao);
	fprintf(L->fp,"\t%d MUTACOES a cada sequencia de %d GERACOES.\n", L->Qtd_Mutacoes_por_vez, L->Qtd_Geracoes_para_Mutacoes);
	fprintf(L->fp,"\tTOTAL de GERACOES: %d.\n\n\n", L->Total_geracoes);
	
	printf("\n\n\tConfiguracao FINALIZADA!!!\n\n");
	
	fclose(L->fp);
	L->fp = fopen("RNA_EVOLUTIVA_RELATORIO_Gabrielly.TXT","a+");
}
//==============================================================================
void geraIndividuos(TLista *L){
	TIndividuo *novo;
	int i, x;
	
    srand( (unsigned)time(NULL) );
    
    for(i= 0; i < L->Qtd_Populacao; i++){
    	novo = (TIndividuo *)malloc(sizeof(TIndividuo));
		
		novo->prox = NULL;
		novo->numero = i+1;		
		novo->erros = -1;
		
		for(x=0; x < MAX_Pesos; x++){
			novo->genes[x] = rand() % 101;
			novo->genes[x] = novo->genes[x] / 100;
		}//for
		
		if(L->populacao == NULL){
			L->populacao = novo;
		} else {
			TIndividuo *atual = L->populacao;
			
			while(atual->prox != NULL){
				atual = atual->prox;
			}//while
			
			atual->prox = novo;
		}//if
		
		L->totalIndividuos++;
	}//for
}
//==============================================================================
void geraLicoes(TLista *L){
	TLicao *novo;
	int p,q;
	
	insereLicao(L, 0, 0, 0);
	insereLicao(L, 0, 1, 0);
	insereLicao(L, 1, 0, 0);
	insereLicao(L, 1, 1, 1);

}
//==============================================================================
void insereLicao(TLista *L, int p, int q, int resultado){
	TLicao *novo = (TLicao *)malloc(sizeof(TLicao));
	
	novo->prox = NULL;
	novo->p = p;
	novo->q = q;
	novo->resultadoEsperado = resultado;
	
	if(L->licoes == NULL){
		L->licoes = novo;
	} else {
		TLicao *atual = L->licoes;
		
		while(atual->prox != NULL){
			atual = atual->prox;			
		}//while
		atual->prox = novo;
	}//if
}
//==============================================================================
void insereNeuronio(TLista *L, int neuronio){
	TNeuronio *novo = (TNeuronio *)malloc(sizeof(TNeuronio));
	novo->prox = NULL;
	novo->neuronio = neuronio;
	novo->peso = 0;
	novo->soma = 0;
	
	if(L->neuronios == NULL){
		L->neuronios = novo;
	} else {
		TNeuronio *atual = L->neuronios;
		
		while(atual->prox != NULL){
			atual = atual->prox;
		}//while
		atual->prox = novo;
	}//if
}
//==============================================================================
void estabelecendoSinapse(TLista *L,int neuronioDe, int neuronioAte, int camada){
	TSinapse *novo = (TSinapse *)malloc(sizeof(TSinapse));
	TSinapse *atual;
	
	novo->prox = NULL;
	novo->neuronio_origem = neuronioDe;
	novo->neuronio_destino = neuronioAte;
	novo->camada = camada;
	novo->peso = 0;
	
	if(L->pesos == NULL){
		L->pesos = novo;
	} else {
		atual = L->pesos;
		
		while(atual->prox != NULL){
			atual = atual->prox;
		}//while
		atual->prox = novo;
	}//if
}
//==============================================================================
void printIndividuos(TLista *L) {
    TIndividuo *atual = L->populacao;
    int i = 1;
    int j = 0;
    fprintf(L->fp, "Lista de indivíduos:\n");
    while (atual != NULL){
        fprintf(L->fp, "Indivíduo: %d, erros: %d\n", i, atual->erros);
        fprintf(L->fp, "Genes: %.2f %.2f %.2f %.2f %.2f %.2f\n",atual->genes[j],atual->genes[j+1],atual->genes[j+2],atual->genes[j+3],atual->genes[j+4],atual->genes[j+5]);
        atual = atual->prox;
        i++;
    }
}
//==============================================================================
void printResultados(TLista *L) {
    fprintf(L->fp, "\n\n=====| Resultados |=====\n\n");
    TIndividuo *indv = L->populacao;
    printf("e: %d", indv->erros);
	
    // Printa o primeiro indivíduo
    TIndividuo *primeiro = L->populacao;
    fprintf(L->fp, "Primeiro Indivíduo:\n");
    fprintf(L->fp, "Número: %d\n", primeiro->numero);
    fprintf(L->fp, "Erros: %d\n", primeiro->erros);
    fprintf(L->fp, "Genes: %.2f %.2f %.2f %.2f %.2f %.2f\n", primeiro->genes[0], primeiro->genes[1], primeiro->genes[2], primeiro->genes[3], primeiro->genes[4], primeiro->genes[5]);

    // Encontra o último indivíduo
    TIndividuo *ultimo = L->populacao;
    while (ultimo->prox != NULL) {
        ultimo = ultimo->prox;
    }

    fprintf(L->fp, "\nÚltimo Indivíduo:\n");
    fprintf(L->fp, "Número: %d\n", ultimo->numero);
    fprintf(L->fp, "Erros: %d\n", ultimo->erros);
    fprintf(L->fp, "Genes: %.2f %.2f %.2f %.2f %.2f %.2f\n", ultimo->genes[0], ultimo->genes[1], ultimo->genes[2], ultimo->genes[3], ultimo->genes[4], ultimo->genes[5]);
}
//==============================================================================
void treinamento(TLista *L){
	printf("\n\n\t\t=====| INICIADO TREINAMENTO |=====\n\n");
	fprintf(L->fp,"\n\n\tINICIO DO TREINAMENTO: ");
	//ponteiro para a struct que armazena data e hora:
	struct tm *data_hora_atual;
	//variável do tipo time_t para armazenar o tempo em segundos.
	time_t segundos;
	//Obetendo o tempo em segundos.
	time(&segundos);
	//Para converter de segundos para o tempo local
	//utilizamos a função localtime().
	data_hora_atual = localtime(&segundos);
	
	fprintf(L->fp,"Dia: %d", data_hora_atual->tm_mday);
	fprintf(L->fp,"   Mes: %d", (data_hora_atual->tm_mon + 1));
	fprintf(L->fp,"   Ano: %d\n\n", (data_hora_atual->tm_year + 1900));
	
	fprintf(L->fp,"Dia da Semana: %d.\n", data_hora_atual->tm_wday);
	
	fprintf(L->fp,"%d", data_hora_atual->tm_hour);
	fprintf(L->fp,":%d", data_hora_atual->tm_min);
	fprintf(L->fp,":%d.\n\n", data_hora_atual->tm_sec);
	
	int i;
	for(i= 0; i < L->Total_geracoes; i++){
		cruzamento(L);
		
		if((i % L->Qtd_Geracoes_para_Mutacoes) == 0){
			promoveMutacoes(L, L->learningRate);
		}//if
		
		avaliacaoIndividuos(L);
		
		ordenamentoIndividuos(L);
		
		poda(L);
	
	}//for
}
//==============================================================================
void insere(TLista *L, TIndividuo *novo) {
    novo->prox = NULL;
    if (L->populacao == NULL) {
        L->populacao = novo;
    } else {
        TIndividuo *atual = L->populacao;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = novo;
    }
    L->totalIndividuos++;
}
//==============================================================================
void cruzamento(TLista *L) {
	/* Função responsável pelo cruzamento de individuos.
	   Cada casal (selecionado por proximidade) gera dois
	   descendentes. E cada descendente herda segmentos
	   do código genético de seus pais.
	*/
	
    TIndividuo *pai1, *pai2, *filho1, *filho2;
    int i, j, metade, cont, indvCruzados, Total;

    pai1 = L->populacao;
    pai2 = pai1->prox;
    cont = L->totalIndividuos + 1;
    indvCruzados = 0; 
    Total = L->totalIndividuos; 

    // Realiza o cruzamento até que metade dos indivíduos tenham sido cruzados
    while (indvCruzados < Total / 2) {
        printf("Cruzando individuo %d com %d\n", pai1->numero, pai2->numero);

        filho1 = (TIndividuo *)malloc(sizeof(TIndividuo));
        filho2 = (TIndividuo *)malloc(sizeof(TIndividuo));
        if (filho1 == NULL || filho2 == NULL) {
            printf("ERRO: Falha na alocação de memória para filhos.\n");
            return;
        }

        filho1->prox = NULL;
        filho1->numero = cont;
        filho1->erros = -1;

        filho2->prox = NULL;
        filho2->numero = cont + 2; 
        filho2->erros = -1;

        // Define o ponto de corte para o cruzamento
        metade = MAX_Pesos / 2;

        // Realiza o cruzamento dos genes entre os pais para gerar os filhos
        for (j = 0; j < metade; j++) {
            filho1->genes[j] = pai1->genes[j];
            filho2->genes[j] = pai2->genes[j];
        }

        for (j = metade; j < MAX_Pesos; j++) {
            filho1->genes[j] = pai2->genes[j];
            filho2->genes[j] = pai1->genes[j];
        }

        cont = cont + 2;
        insere(L, filho1);
        insere(L, filho2);
        pai1 = pai2->prox;

        // Verifica se chegou ao final da lista, retornando aos primeiros pais
        if (pai1 != L->populacao) {
            pai2 = pai1->prox;
        } else {
            // Se pai1 e pai2 são os primeiros elementos da lista, sai do loop
            if (pai1 == L->populacao && pai2 == L->populacao->prox) {
                break; 
            }
        }
        indvCruzados++;
    }
}

//==============================================================================
void avaliacaoIndividuos(TLista *L){
	/*
	Avalia o grau de adaptação de cada indivíduo ao ambiente
	em termos de quantidade de erros cometidos nas lições da 
	RNA. O objetivo é MINIMIZAR esses ERROS até ZERO.
	*/
	
    TIndividuo *indv = L->populacao;
    int p, q, resultadoEsperado, resultadoObtido;
    float n1, n2, n3, n4, n5, soma3, soma4, soma5;
    float peso13, peso14, peso23, peso24, peso35, peso45;

    while(indv != NULL){
        TLicao *licao = L->licoes; 

        printf("Avaliando indivíduo número %d\n", indv->numero);

        indv->erros = 0; // Inicializa o número de erros do indivíduo com 0

        while (licao != NULL) {
            printf("Processando lição... P: %d, Q: %d, \tResultado Esperado: %d \t", licao->p, licao->q, licao->resultadoEsperado);

            n1 = licao->p; 
            n2 = licao->q;

            resultadoEsperado = licao->resultadoEsperado;

            peso13 = indv->genes[0];
            peso14 = indv->genes[1];
            peso23 = indv->genes[2];
            peso24 = indv->genes[3];
            peso35 = indv->genes[4];
            peso45 = indv->genes[5];
            
            soma3 = (n1*peso13) + (n2*peso23);
            if(soma3 >= L->sinapseThreshold){
                n3 = 1;
            } else{
                n3 = 0;
            }
            
            soma4 = (n1*peso14) + (n2*peso24);
            if(soma4 >= L->sinapseThreshold){
                n4 = 1;
            } else {
                n4 = 0;
            }
            
            soma5 = (n3*peso35) + (n4*peso45);
            if(soma5 >= L->sinapseThreshold){
                n5 = 1;
            } else{
                n5 = 0;
            }
            
            resultadoObtido = n5;

            printf("Resultado Obtido: %d\n", resultadoObtido);

            if(resultadoEsperado != resultadoObtido){
                printf("Cometeu erro\n");  
                indv->erros++; 
            }
            
            licao = licao->prox;    
        }

        printf("Número de erros do indivíduo %d: %d\n", indv->numero, indv->erros);

        indv = indv->prox;
    }
}
//==============================================================================
void ordenamentoIndividuos(TLista *L){
	/* Reordena os indivíduos por ordem ascendente de erros:
	   os indivíduos que cometeram menos erros deverão permanecer
	   no início da Lista e os que cometeram mais erros deverão 
	   ficar no final da mesma Lista. */
	   
	TIndividuo *atual, *anterior, *aux;
	int trocou;
	
	if (L->populacao == NULL || L->populacao->prox == NULL) {
        return;
    }
	
	do{
		trocou = 0;
		anterior = NULL;
		atual = L->populacao;
		
		while(atual->prox != NULL){
			if(atual->erros > atual->prox->erros){
				aux = atual->prox;
                atual->prox= atual->prox->prox;
                aux->prox = atual;
                
				if(anterior != NULL){
					anterior->prox = aux;
				} else{
					L->populacao = aux;
				}
				anterior = aux;
                trocou = 1;	
			} else{
				anterior = atual;
            	atual = atual->prox;
        	}
		}// while
	} while(trocou != 0);	   
}
//==============================================================================
void promoveMutacoes(TLista *L, float learningRate) {
	/* Altera o código genético de um número específico
	   de indivíduos (= L->Qtd_Mutacoes_por_vez). */
	   
	int indvEscolhido, cont, gene, direcaoMutacao, i;
	
    if (L->populacao == NULL) {
        printf("Lista de individuos vazia.\n");
        return;
    }

    TIndividuo *individuo = L->populacao;
    indvEscolhido = rand() % L->totalIndividuos; 
    cont = 0;

    while (individuo != NULL) {
        // Verificar se este é o indivíduo escolhido para a mutação
        if (cont == indvEscolhido) {
            printf("Mutação no indivíduo %d\n", individuo->numero);

            for (i = 0; i < L->Qtd_Mutacoes_por_vez; i++) {
                gene = rand() % MAX_Pesos;

                // Escolher aleatoriamente se a mutação será para cima ou para baixo
                direcaoMutacao = rand() % 2; // 0 para baixo, 1 para cima

                if (direcaoMutacao == 0) {
                    individuo->genes[gene] = individuo->genes[gene] - learningRate; // Diminuir do valor do gene
                } else {
                    individuo->genes[gene] = individuo->genes[gene] + learningRate; // Aumentar o valor do gene
                }
            }
            break;
        }
        individuo = individuo->prox;
        cont++;
    }
    printf("Mutacao promovida com sucesso.\n");
}

//==============================================================================
void poda(TLista *L){
	/* Elimina os indivíduos menos aptos (que estão no
	   fim da Lista) até que a população volte ao seu
	   Limite estabelecido na configuração inicial 
	   (L->Qtd_Populacao). */

    TIndividuo *anterior, *atual;
    int i;
    
    while(L->totalIndividuos > L->Qtd_Populacao){
        anterior = NULL;
        atual = L->populacao;
        
		// Definir o anterior 
        for(i = 0; i < L->Qtd_Populacao - 1; i++){
            anterior = atual;
            atual = atual->prox;
        }

        // Se houver algum nó anterior, significa que precisamos cortar a lista
        if (anterior!= NULL) {

            // Definir o próximo após o último nó como NULL
            anterior->prox = NULL;

            // Atualizar o número total de indivíduos na lista
            L->totalIndividuos = L->Qtd_Populacao;
        } else {
            // Se não houver nenhum anterior, significa que a lista inteira deve ser removida
            L->populacao = NULL;
            L->totalIndividuos = 0;
        }
    }
	printResultados(L);
	printIndividuos(L);
}
//==============================================================================