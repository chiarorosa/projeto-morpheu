// Texts to read aloud when recording a reference voice.
//
// Written for cloning, not for meaning: short sentences that each end in clear
// punctuation (the reference is cut between sentences, so every sentence is a
// usable window), every vowel and the consonants that give an accent away, and
// a statement, a question and an exclamation so the model hears the speaker's
// whole intonation range. Numbers are written out so what is read is exactly
// what is typed. Read at a normal pace they run 25-35 s.

export interface RecordingScript {
  id: string;
  label: string;
  hint: string;
  text: string;
}

const PORTUGUESE: RecordingScript[] = [
  {
    id: "pt-narracao",
    label: "Narração",
    hint: "Leia num tom calmo, como quem conta uma história.",
    text:
      "Olá! Hoje o céu amanheceu limpo, e o vento trouxe cheiro de terra molhada lá do horizonte. " +
      "Você acredita que a vizinha ganhou um jogo de xadrez em junho? " +
      "Pois é, a vida às vezes surpreende a gente. " +
      "Enquanto isso, o velho relojoeiro consertava os ponteiros com calma, assobiando uma canção antiga. " +
      "Amanhã, bem cedo, vamos caminhar juntos até a praça.",
  },
  {
    id: "pt-conversa",
    label: "Conversa",
    hint: "Leia como se falasse com um amigo: natural, com sorriso na voz.",
    text:
      "Ei, tudo bem com você? " +
      "Ontem fui à feira e comprei três quilos de laranja, um abacaxi enorme e duas dúzias de ovos. " +
      "O rapaz do caixa errou o troco e ainda reclamou comigo! " +
      "No fim, a gente riu bastante, e eu voltei pra casa assobiando. " +
      "Me conta: e o seu fim de semana, como foi?",
  },
  {
    id: "pt-apresentacao",
    label: "Apresentação",
    hint: "Leia com clareza e segurança, como num vídeo ou aula.",
    text:
      "Bem-vindos a mais um episódio. " +
      "Hoje vamos falar sobre uma ideia simples, mas que muda tudo: aprender um pouco a cada dia. " +
      "Parece pouco? Em um ano, são trezentas e sessenta e cinco pequenas vitórias. " +
      "Então pegue um café, fique à vontade e venha comigo. " +
      "Garanto que, no final, você vai enxergar o seu trabalho de outro jeito.",
  },
];

const ENGLISH: RecordingScript[] = [
  {
    id: "en-narration",
    label: "Narration",
    hint: "Read calmly, as if telling a story.",
    text:
      "Hello! The morning light crept over the hills, and the air smelled of fresh rain. " +
      "Did you know that my neighbor won a chess tournament last June? " +
      "Well, life has a funny way of surprising us. " +
      "Meanwhile, the old watchmaker quietly fixed the clock, whistling a cheerful tune. " +
      "Tomorrow, just after sunrise, we'll walk together to the village square.",
  },
];

const SPANISH: RecordingScript[] = [
  {
    id: "es-narracion",
    label: "Narración",
    hint: "Lee con calma, como quien cuenta una historia.",
    text:
      "¡Hola! Hoy el cielo amaneció despejado, y el viento trajo olor a tierra mojada desde el horizonte. " +
      "¿Sabías que mi vecina ganó un torneo de ajedrez en junio? " +
      "Pues sí, la vida a veces nos sorprende. " +
      "Mientras tanto, el viejo relojero arreglaba las agujas con calma, silbando una canción antigua. " +
      "Mañana, muy temprano, caminaremos juntos hasta la plaza.",
  },
];

export const RECORDING_SCRIPTS: Record<string, RecordingScript[]> = {
  Portuguese: PORTUGUESE,
  English: ENGLISH,
  Spanish: SPANISH,
};

export function scriptsFor(language: string): RecordingScript[] {
  return RECORDING_SCRIPTS[language] ?? PORTUGUESE;
}

export const RECORDING_TIPS = [
  "Cômodo silencioso, sem ventilador ou ar-condicionado ligado.",
  "Microfone a um palmo da boca, um pouco de lado.",
  "Fale do jeito que quer que a voz clonada soe: o modelo copia ritmo e emoção.",
  "Leia exatamente o texto, com as pausas da pontuação. Errou? Grave de novo.",
];
