/**
 * Chat com IA local usando a Prompt API do Chrome (Gemini Nano).
 *
 * O modelo roda inteiramente no navegador, sem servidor. O fluxo é:
 *   1. main() verifica se o navegador suporta a API e se o modelo está baixado;
 *   2. configura os sliders de temperatura e topK com os limites do modelo;
 *   3. a cada pergunta, cria uma sessão nova e exibe a resposta em streaming,
 *      permitindo interromper a geração pelo botão "Parar".
 */

// ============================================================================
// Estado e referências do DOM
// ============================================================================

// Estado compartilhado da geração atual
const aiContext = {
    session: null,          // sessão ativa do LanguageModel (recriada a cada pergunta)
    abortController: null,  // permite cancelar o streaming em andamento
    isGenerating: false,    // true enquanto uma resposta está sendo gerada
};

const elements = {
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temp-value'),
    topK: document.getElementById('topK'),
    topKValue: document.getElementById('topk-value'),
    form: document.getElementById('question-form'),
    questionInput: document.getElementById('question'),
    output: document.getElementById('output'),
    button: document.getElementById('ask-button'),
    year: document.getElementById('year'),
};

// Idioma esperado na entrada; usado tanto na verificação quanto na criação da sessão
const LANGUAGE = 'pt';

// Valores usados quando LanguageModel.params() não está disponível.
// Nas versões recentes do Chrome, params() foi removido do contexto web
// (continua existindo apenas em extensões).
const DEFAULT_PARAMS = {
    defaultTemperature: 1,
    defaultTopK: 3,
    maxTemperature: 2,
    maxTopK: 128,
};

const SYSTEM_PROMPT = `
    Você é um assistente de IA que responde de forma clara e objetiva.
    Responda sempre em formato de texto ao invés de markdown`;

// ============================================================================
// Verificação de requisitos
// ============================================================================

/**
 * Confere se o navegador suporta a Prompt API e se o modelo está pronto.
 * Se o modelo ainda não foi baixado, dispara o download.
 *
 * @returns {Promise<string[] | null>} lista de mensagens de erro, ou null se tudo estiver ok
 */
async function checkRequirements() {
    const errors = [];
    const returnResults = () => errors.length ? errors : null;

    // @ts-ignore
    const isChrome = !!window.chrome;
    if (!isChrome)
        errors.push("⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).");

    // Sem a API global não há o que fazer: a flag precisa ser habilitada
    if (!('LanguageModel' in self)) {
        errors.push("⚠️ As APIs nativas de IA não estão ativas.");
        errors.push("Ative a seguinte flag em chrome://flags/:");
        errors.push("- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)");
        errors.push("Depois reinicie o Chrome e tente novamente.");
        return returnResults();
    }

    // Possíveis valores: 'available' | 'unavailable' | 'downloading' | 'downloadable'
    const availability = await LanguageModel.availability({ languages: [LANGUAGE] });
    console.log('Language Model Availability:', availability);
    if (availability === 'available') {
        return returnResults();
    }

    if (availability === 'unavailable') {
        errors.push(`⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`);
    }

    if (availability === 'downloading') {
        errors.push(`⚠️ O modelo de linguagem de IA está sendo baixado. Por favor, aguarde alguns minutos e tente novamente.`);
    }

    if (availability === 'downloadable') {
        errors.push(`⚠️ O modelo de linguagem de IA precisa ser baixado, baixando agora... (acompanhe o progresso no terminal do chrome)`);
        try {
            // Criar uma sessão é o que dispara o download do modelo.
            // O monitor permite acompanhar o progresso no console.
            const session = await LanguageModel.create({
                expectedInputLanguages: [LANGUAGE],
                monitor(m) {
                    m.addEventListener('downloadprogress', (e) => {
                        const percent = ((e.loaded / e.total) * 100).toFixed(0);
                        console.log(`Downloaded ${percent}%`);
                    });
                }
            });
            // Um prompt simples garante que o modelo foi carregado por completo
            await session.prompt('Olá');
            session.destroy();

            // Verifica novamente após o download
            const newAvailability = await LanguageModel.availability({ languages: [LANGUAGE] });
            if (newAvailability === 'available') {
                return null; // download concluído com sucesso
            }
        } catch (error) {
            console.error('Error downloading model:', error);
            errors.push(`⚠️ Erro ao baixar o modelo: ${error.message}`);
        }
    }

    return returnResults();
}

// ============================================================================
// Interação com o modelo
// ============================================================================

/**
 * Envia a pergunta ao modelo e devolve a resposta aos poucos (streaming).
 * Generator assíncrono: cada `yield` entrega um pedaço do texto gerado.
 *
 * @param {string} question pergunta do usuário
 * @param {number} temperature aleatoriedade da resposta (0 = mais determinística)
 * @param {number} topK quantidade de tokens candidatos considerados a cada passo
 */
async function* askAI(question, temperature, topK) {
    // Cancela qualquer geração anterior e prepara um novo controlador
    aiContext.abortController?.abort();
    aiContext.abortController = new AbortController();

    // Temperatura e topK só podem ser definidos na criação da sessão,
    // por isso a sessão anterior é descartada e uma nova é criada
    if (aiContext.session) {
        aiContext.session.destroy();
    }

    const session = await LanguageModel.create({
        expectedInputLanguages: [LANGUAGE],
        temperature: temperature,
        topK: topK,
        initialPrompts: [
            { role: 'system', content: SYSTEM_PROMPT },
        ],
    });

    const responseStream = await session.promptStreaming(
        [
            { role: 'user', content: question },
        ],
        {
            // Ao abortar o controlador, o modelo para de gerar
            signal: aiContext.abortController.signal,
        }
    );

    for await (const chunk of responseStream) {
        if (aiContext.abortController.signal.aborted) {
            break;
        }
        yield chunk;
    }
}

// ============================================================================
// Interface (UI)
// ============================================================================

/**
 * Alterna o botão entre "Enviar" e "Parar".
 * Ao voltar para "Enviar", aborta a geração em andamento (se houver).
 *
 * @param {boolean} isGenerating true para entrar no modo "Parar"
 */
function toggleSendOrStopButton(isGenerating) {
    if (isGenerating) {
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Parar';
        elements.button.classList.add('stop-button');
    } else {
        aiContext.abortController?.abort();
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Enviar';
        elements.button.classList.remove('stop-button');
    }
}

/**
 * Lê a pergunta e os parâmetros do formulário, chama o modelo e
 * escreve a resposta na tela conforme os pedaços chegam.
 */
async function onSubmitQuestion() {
    const questionInput = elements.questionInput;
    const output = elements.output;
    const question = questionInput.value;

    if (!question.trim()) {
        return;
    }

    // Parâmetros escolhidos nos sliders
    const temperature = parseFloat(elements.temperature.value);
    const topK = parseInt(elements.topK.value);
    console.log('Using parameters:', { temperature, topK });

    toggleSendOrStopButton(true);

    output.textContent = 'Processing your question...';
    const aiResponseChunks = askAI(question, temperature, topK);
    output.textContent = '';

    // Exibe cada pedaço da resposta assim que ele chega
    for await (const chunk of aiResponseChunks) {
        if (aiContext.abortController.signal.aborted) {
            break;
        }
        console.log('Received chunk:', chunk);
        output.textContent += chunk;
    }

    toggleSendOrStopButton(false);
}

/**
 * Ajusta os sliders de temperatura e topK conforme os limites do modelo.
 *
 * @param {typeof DEFAULT_PARAMS} params limites e valores padrão do modelo
 */
function setupSliders(params) {
    elements.topK.min = 1;
    elements.topK.max = params.maxTopK;
    elements.topK.value = params.defaultTopK;
    elements.topKValue.textContent = params.defaultTopK;

    elements.temperature.min = 0;
    elements.temperature.max = params.maxTemperature;
    elements.temperature.value = params.defaultTemperature;
    elements.temperatureValue.textContent = params.defaultTemperature;
}

function setupEventListeners() {
    // Mantém o valor exibido ao lado de cada slider atualizado
    elements.temperature.addEventListener('input', (e) => {
        elements.temperatureValue.textContent = e.target.value;
    });

    elements.topK.addEventListener('input', (e) => {
        elements.topKValue.textContent = e.target.value;
    });

    // O mesmo botão envia a pergunta ou, durante a geração, interrompe a resposta
    elements.form.addEventListener('submit', async function (event) {
        event.preventDefault();

        if (aiContext.isGenerating) {
            toggleSendOrStopButton(false);
            return;
        }

        onSubmitQuestion();
    });
}

// ============================================================================
// Inicialização
// ============================================================================

(async function main() {
    elements.year.textContent = new Date().getFullYear();

    // Se algum requisito falhar, mostra os erros e desabilita o envio
    const reqErrors = await checkRequirements();
    if (reqErrors) {
        elements.output.innerHTML = reqErrors.join('<br/>');
        elements.button.disabled = true;
        return;
    }

    // Usa os limites reais do modelo quando disponíveis; senão, os valores padrão
    const params = typeof LanguageModel.params === 'function'
        ? await LanguageModel.params()
        : DEFAULT_PARAMS;
    console.log('Language Model Params:', params);

    setupSliders(params);
    setupEventListeners();
})();
