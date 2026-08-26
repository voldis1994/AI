# Mans AI — Personīgais asistents

Tavs personīgais mākslīgā intelekta asistents ar tērzēšanas saskarni, atmiņu un pilnu personalizāciju.

![Next.js](https://img.shields.io/badge/Next.js-15-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-38bdf8)

## Funkcijas

- **Tērzēšana reāllaikā** — straumētas atbildes no OpenAI modeļiem
- **Personalizācija** — nosaki savu vārdu, AI vārdu un personības tipu
- **4 personības** — Asistents, Draugs, Skolotājs, Radošais
- **Atmiņa** — AI atceras svarīgus faktus par tevi visās sarunās
- **Sarunu vēsture** — vairākas sarunas, kas saglabājas pārlūkprogrammā
- **Latviešu un angļu valoda** — saskarne un atbildes abās valodās

## Ātrais starts

### 1. Instalē atkarības

```bash
npm install
```

### 2. Iestati API atslēgu

Izveido `.env.local` failu (skatīt `.env.example`):

```bash
cp .env.example .env.local
```

Pievieno savu OpenAI API atslēgu:

```
OPENAI_API_KEY=sk-tava-atslega
OPENAI_MODEL=gpt-4o-mini
```

API atslēgu vari iegūt: https://platform.openai.com/api-keys

### 3. Palaid lietotni

```bash
npm run dev
```

Atver pārlūkprogrammā: http://localhost:3000

## Iestatījumi

Atver **Iestatījumi** paneli, lai konfigurētu:

| Lauks | Apraksts |
|-------|----------|
| Vārds | Tavs vārds — AI tevi tā sauks |
| AI vārds | Asistenta vārds (noklusējums: "Mans AI") |
| Personība | Kā AI komunicē ar tevi |
| Par sevi | Īss apraksts par tevi |
| Intereses | Tavas intereses |
| Atmiņa | Fakti, ko AI atceras visās sarunās |

## Personības tipi

- **Asistents** — profesionāls un efektīvs ikdienas palīgs
- **Draugs** — siltš un draudzīgs sarunu biedrs
- **Skolotājs** — pacietīgs skaidrotājs soli pa solim
- **Radošais** — iedomīgs partneris idejām un projektiem

## Tehnoloģijas

- [Next.js 15](https://nextjs.org/) — React framework
- [OpenAI API](https://platform.openai.com/) — AI modelis
- [Tailwind CSS](https://tailwindcss.com/) — stili
- [Lucide React](https://lucide.dev/) — ikonas

## Izstrāde

```bash
npm run dev    # izstrādes serveris
npm run build  # produkcijas builds
npm run start  # produkcijas serveris
npm run lint   # koda pārbaude
```

## Datu glabāšana

Visi iestatījumi, sarunas un atmiņa tiek glabāta lokāli tavā pārlūkprogrammā (`localStorage`). Nekādi dati netiek sūtīti uz ārējiem serveriem, izņemot sarunas ar OpenAI API.

## Licence

MIT
