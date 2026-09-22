const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_4x3"; // 10 x 7.5"
p.author = "Рубцов С.А.";

// ---- палитра «сталь + жар» ----
const DARK="1F2A38", HEAT="E1442F", STEEL="2E6C99", GREEN="2FA36B",
      RED="D63B2A", AMBER="E8A33D", LIGHT="F5F7FA", MUTED="5A6B7B",
      WHITE="FFFFFF", INK="22303F", CARD="FFFFFF";
const IMG = "C:/Users/arena/Desktop/Programmer/Диссертация/magnet-platform/docs/conferences/";
// Рисунки, собираемые генераторами из репозитория (числа считаются, а не вписаны руками)
const FIG = "C:/Users/arena/Desktop/Programmer/Диссертация/magnet-platform/docs/papers/experiments/figs/";
const TB = "Arial Black", BODY = "Arial";
const W = 10, H = 7.5, M = 0.55;

function bg(s, c){ s.background = { color: c }; }
function title(s, t, color){
  s.addText(t, { x:M, y:0.38, w:W-2*M, h:0.9, fontFace:TB, fontSize:26, bold:true,
    color: color||DARK, align:"left", valign:"middle", margin:0 });
  // мотив: жирная точка-«магнит» слева от области контента (не полоса)
  s.addShape(p.ShapeType.ellipse, { x:M, y:1.34, w:0.16, h:0.16, fill:{color:HEAT} });
}
function card(s, x, y, w, h, fill){
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius:0.08,
    fill:{color:fill||CARD}, line:{color:"E4E9EF", width:1},
    shadow:{ type:"outer", color:"9AA7B4", blur:6, offset:2, angle:90, opacity:0.28 } });
}
function badge(s, x, y, txt, col){
  s.addShape(p.ShapeType.ellipse, { x, y, w:0.62, h:0.62, fill:{color:col} });
  s.addText(txt, { x, y, w:0.62, h:0.62, fontFace:TB, fontSize:20, bold:true, color:WHITE, align:"center", valign:"middle", margin:0 });
}
function arrow(s, x1,y1,x2,y2, col){
  s.addShape(p.ShapeType.line, { x:Math.min(x1,x2), y:Math.min(y1,y2),
    w:Math.abs(x2-x1), h:Math.abs(y2-y1),
    line:{ color:col, width:3, endArrowType:"triangle", beginArrowType:"none" },
    flipH: x2<x1, flipV: y2<y1 });
}

// ============================ S1 · ТИТУЛ ============================
{ const s=p.addSlide(); bg(s,DARK);
  s.addText("Расчётный метод оценки стойкости постоянных магнитов к тепловому размагничиванию и обоснования выбора материала в электродвигателях БПЛА",
    { x:M, y:1.5, w:W-2*M, h:2.6, fontFace:TB, fontSize:27, bold:true, color:WHITE, align:"left", valign:"top", lineSpacingMultiple:1.02 });
  s.addShape(p.ShapeType.ellipse,{x:M,y:4.35,w:0.18,h:0.18,fill:{color:HEAT}});
  s.addShape(p.ShapeType.ellipse,{x:M+0.28,y:4.35,w:0.18,h:0.18,fill:{color:AMBER}});
  s.addShape(p.ShapeType.ellipse,{x:M+0.56,y:4.35,w:0.18,h:0.18,fill:{color:GREEN}});
  s.addText([
    {text:"Рубцов С. А.", options:{bold:true, color:WHITE, fontSize:20, breakLine:true}},
    {text:"инженер 1 категории", options:{color:"C7D2DC", fontSize:15, breakLine:true}},
    {text:"АО «НПП «Исток» им. А. И. Шокина», г. Фрязино", options:{color:"C7D2DC", fontSize:15}},
  ], { x:M, y:4.75, w:W-2*M, h:1.2, fontFace:BODY, align:"left", valign:"top", lineSpacingMultiple:1.1 });
  s.addText("III Всероссийская НТК «Постоянные магниты: наука и технологии, производство, применение»   ·   Москва, 6–7 октября 2026",
    { x:M, y:6.75, w:W-2*M, h:0.5, fontFace:BODY, fontSize:12, color:"8FA0AE", align:"left" });
  s.addNotes("Здравствуйте. Тема доклада — расчётный метод, который в реальных рабочих условиях двигателя оценивает необратимое тепловое размагничивание магнита и помогает обосновать выбор материала.");
}

// ============================ S2 · ПРОБЛЕМА ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Проблема: магнит работает у предела");
  s.addText("В теплонагруженном двигателе БПЛА на магнит действуют два фактора, способствующих размагничиванию:",
    { x:M, y:1.5, w:W-2*M, h:1.0, fontFace:BODY, fontSize:19, color:INK, align:"left", valign:"top", lineSpacingMultiple:1.05 });
  const cy=2.7, cw=(W-2*M-0.4)/2, ch=2.5;
  card(s,M,cy,cw,ch);
  s.addShape(p.ShapeType.ellipse,{x:M+0.35,y:cy+0.28,w:0.66,h:0.66,fill:{color:HEAT}});
  s.addText("🔥",{x:M+0.35,y:cy+0.28,w:0.66,h:0.66,fontSize:24,align:"center",valign:"middle",margin:0});
  s.addText("Высокая температура",{x:M+0.3,y:cy+1.02,w:cw-0.6,h:0.45,fontFace:TB,fontSize:18,bold:true,color:HEAT,align:"left",margin:0});
  s.addText("нагрев поднимает кривую размагничивания: колено приближается к рабочей точке",{x:M+0.3,y:cy+1.46,w:cw-0.6,h:0.95,fontFace:BODY,fontSize:14,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.05});
  const x2=M+cw+0.4; card(s,x2,cy,cw,ch);
  s.addShape(p.ShapeType.ellipse,{x:x2+0.35,y:cy+0.28,w:0.66,h:0.66,fill:{color:STEEL}});
  s.addText("⚡",{x:x2+0.35,y:cy+0.28,w:0.66,h:0.66,fontSize:22,align:"center",valign:"middle",margin:0});
  s.addText("Поле реакции якоря",{x:x2+0.3,y:cy+1.02,w:cw-0.6,h:0.45,fontFace:TB,fontSize:18,bold:true,color:STEEL,align:"left",margin:0});
  s.addText("ток обмотки создаёт встречное, размагничивающее магнитное поле",{x:x2+0.3,y:cy+1.46,w:cw-0.6,h:0.95,fontFace:BODY,fontSize:14,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.05});
  card(s,M,5.45,W-2*M,1.05,"FDECEA");
  s.addText([{text:"Итог: ",options:{bold:true,color:RED}},{text:"риск необратимой потери намагниченности — двигатель безвозвратно теряет момент.",options:{color:INK}}],
    {x:M+0.3,y:5.45,w:W-2*M-0.6,h:1.05,fontFace:BODY,fontSize:18,align:"left",valign:"middle",margin:0});
  s.addNotes("Магнит зажат между нагревом и полем реакции якоря. Оба толкают его к размагничиванию, и часть потери — необратимая.");
}

// ============================ S3 · КОЛЕНО (KEY) ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Необратимое размагничивание: что происходит");
  s.addImage({ path:FIG+"fig_knee_loadline_slide.png", x:M, y:1.55, w:5.9, h:4.6 });
  const rx=6.7, rw=W-M-rx;
  const items=[
    ["1","Холодный магнит: рабочая точка далеко от колена.",GREEN],
    ["2","Нагрев двигает не точку, а КОЛЕНО — запас падает в разы.",AMBER],
    ["3","Реакция якоря уводит точку за колено — потеря навсегда.",RED],
  ];
  let yy=2.0;
  items.forEach(([n,t,c])=>{
    s.addShape(p.ShapeType.ellipse,{x:rx,y:yy,w:0.5,h:0.5,fill:{color:c}});
    s.addText(n,{x:rx,y:yy,w:0.5,h:0.5,fontFace:TB,fontSize:17,bold:true,color:WHITE,align:"center",valign:"middle",margin:0});
    s.addText(t,{x:rx+0.65,y:yy-0.05,w:rw-0.65,h:1.1,fontFace:BODY,fontSize:16.5,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.05});
    yy+=1.4;
  });
  s.addNotes("Ключевой слайд. Рабочая точка магнита лежит на нагрузочной прямой — её наклон задан геометрией: толщиной магнита и зазором. При нагреве точка съезжает по ЭТОЙ ЖЕ прямой вниз, и размагничивающее поле в магните даже уменьшается — с 201 до 171 килоампера на метр. Но колено идёт к нулю впятеро быстрее: за 130 градусов оно уезжает с 1353 до 386. То есть за колено магнит уходит не потому, что поле выросло, а потому что колено догнало точку — запас падает с 1152 до 215. Одного нагрева ещё мало. Добивает ток: реакция якоря при перегрузе сдвигает нагрузочную прямую влево, точка уходит за колено — и вот это уже необратимая потеря.");
}

// ============================ S4 · МОТИВАЦИЯ ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Как защищаются сейчас — и чего не хватает");
  const cy=1.7, cw=(W-2*M-0.4)/2, ch=2.15;
  card(s,M,cy,cw,ch);
  s.addText("Запас по коэрцитивной силе",{x:M+0.3,y:cy+0.28,w:cw-0.6,h:0.5,fontFace:TB,fontSize:17,bold:true,color:INK,align:"left",margin:0});
  s.addText([{text:"требует тяжёлых редкоземельных элементов ",options:{color:INK}},{text:"(Dy, Tb)",options:{bold:true,color:HEAT}},{text:" — дефицитных и импортозависимых",options:{color:INK}}],
    {x:M+0.3,y:cy+0.85,w:cw-0.6,h:1.1,fontFace:BODY,fontSize:15.5,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.08});
  const x2=M+cw+0.4; card(s,x2,cy,cw,ch);
  s.addText("Увеличение габаритов",{x:x2+0.3,y:cy+0.28,w:cw-0.6,h:0.5,fontFace:TB,fontSize:17,bold:true,color:INK,align:"left",margin:0});
  s.addText("двигатель становится тяжелее и дороже — критично для летательного аппарата",
    {x:x2+0.3,y:cy+0.85,w:cw-0.6,h:1.1,fontFace:BODY,fontSize:15.5,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.08});
  card(s,M,4.35,W-2*M,1.85,"EAF1F7");
  s.addText("Чего нет:",{x:M+0.35,y:4.6,w:W-2*M-0.7,h:0.5,fontFace:TB,fontSize:18,bold:true,color:STEEL,align:"left",margin:0});
  s.addText("расчётного метода, который даёт вердикт о необратимом размагничивании в реальных условиях машины и обосновывает выбор материала.",
    {x:M+0.35,y:5.1,w:W-2*M-0.7,h:1.0,fontFace:BODY,fontSize:18,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.08});
  s.addNotes("Сегодня либо переплачивают тяжёлыми РЗМ, либо переразмеривают машину. Инструмента, дающего вердикт в рабочих условиях, нет — это и есть ниша работы.");
}

// ============================ S5 · ИДЕЯ (петля) ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Идея: считать поле, тепло и размагничивание совместно");
  // 3 узла в треугольнике
  const nodes=[
    {x:3.55,y:1.9,w:2.9,h:1.0,c:STEEL,t:"Электромагнитный расчёт\n(поле, потери)"},
    {x:5.9,y:4.55,w:2.9,h:1.05,c:HEAT,t:"Тепловой расчёт\n(температура растёт)"},
    {x:1.2,y:4.55,w:2.9,h:1.05,c:GREEN,t:"Магнит слабеет:\nB(H,T), необратимость"},
  ];
  nodes.forEach(n=>{ card(s,n.x,n.y,n.w,n.h);
    s.addText(n.t,{x:n.x+0.12,y:n.y,w:n.w-0.24,h:n.h,fontFace:BODY,fontSize:14.5,bold:true,color:n.c,align:"center",valign:"middle",margin:0,lineSpacingMultiple:1.0}); });
  // стрелки по кругу
  arrow(s, 5.6,2.6, 6.9,4.5, HEAT);   // A -> B  (потери -> нагрев)
  arrow(s, 5.7,5.6, 4.0,5.6, RED);    // B -> C  (температура -> свойства)
  arrow(s, 2.5,4.5, 3.9,2.9, GREEN);  // C -> A  (новое поле)
  s.addText("потери → нагрев",{x:6.35,y:3.35,w:2.0,h:0.4,fontFace:BODY,fontSize:12,italic:true,color:MUTED,align:"left",margin:0});
  s.addText("температура → свойства",{x:3.9,y:5.72,w:2.3,h:0.4,fontFace:BODY,fontSize:12,italic:true,color:MUTED,align:"center",margin:0});
  s.addText("новое поле",{x:1.55,y:3.35,w:1.8,h:0.4,fontFace:BODY,fontSize:12,italic:true,color:MUTED,align:"left",margin:0});
  s.addText("единая связанная задача",{x:3.9,y:3.55,w:2.2,h:0.6,fontFace:TB,fontSize:14,bold:true,color:DARK,align:"center",valign:"middle",margin:0});
  s.addText("Всё влияет друг на друга, поэтому решается совместно — до установившегося теплового состояния.",
    {x:M,y:6.5,w:W-2*M,h:0.7,fontFace:BODY,fontSize:16,color:INK,align:"center",valign:"middle",lineSpacingMultiple:1.03});
  s.addNotes("Поле даёт потери, потери греют, температура меняет магнит, магнит меняет поле. Круг замкнут — считаем как одну связанную задачу до выхода на установившуюся температуру.");
}

// ============================ S6 · КАК СЧИТАЕМ ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Как устроен расчёт");
  const rows=[
    ["МКЭ","Метод конечных элементов в двумерной постановке.",STEEL],
    ["B(T)","Магнит — модель B(H,T): по параметрам и температурным коэффициентам строит кривую при любой температуре.",HEAT],
    ["↩","Необратимость — по линии возврата: запоминается наиболее неблагоприятная точка за историю нагружения.",GREEN],
    ["⏱","Считаем по времени до установившейся температуры; дополнительно распознаём тепловой разгон.",AMBER],
  ];
  let yy=1.75; const rh=1.12;
  rows.forEach(([ic,t,c])=>{
    card(s,M,yy,W-2*M,rh-0.15);
    s.addShape(p.ShapeType.ellipse,{x:M+0.25,y:yy+0.2,w:0.58,h:0.58,fill:{color:c}});
    s.addText(ic,{x:M+0.25,y:yy+0.2,w:0.58,h:0.58,fontFace:BODY,fontSize:14,bold:true,color:WHITE,align:"center",valign:"middle",margin:0});
    s.addText(t,{x:M+1.1,y:yy,w:W-2*M-1.4,h:rh-0.15,fontFace:BODY,fontSize:16.5,color:INK,align:"left",valign:"middle",margin:0,lineSpacingMultiple:1.03});
    yy+=rh;
  });
  s.addNotes("Четыре опоры метода: МКЭ 2D; модель магнита, которая сама рассчитывает кривую при нужной температуре; учёт необратимости по линии возврата; счёт по времени до установившегося состояния с распознаванием разгона.");
}

// ============================ S7 · ДОСТОВЕРНОСТЬ ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Достоверность — без подгонки");
  const items=[
    ["Точное решение","Сравнение с задачами, имеющими точное аналитическое решение."],
    ["Баланс энергии","Контроль энергетического баланса в ходе расчёта."],
    ["Реальный двигатель","Проверка на двигателе с паспортными данными."],
  ];
  const cw=(W-2*M-0.8)/3, cy=1.85, ch=3.0;
  items.forEach(([h1,t],i)=>{ const x=M+i*(cw+0.4);
    card(s,x,cy,cw,ch);
    s.addShape(p.ShapeType.ellipse,{x:x+cw/2-0.42,y:cy+0.35,w:0.84,h:0.84,fill:{color:GREEN}});
    s.addText("✓",{x:x+cw/2-0.42,y:cy+0.35,w:0.84,h:0.84,fontFace:TB,fontSize:30,bold:true,color:WHITE,align:"center",valign:"middle",margin:0});
    s.addText(h1,{x:x+0.2,y:cy+1.35,w:cw-0.4,h:0.6,fontFace:TB,fontSize:16,bold:true,color:INK,align:"center",valign:"middle",margin:0});
    s.addText(t,{x:x+0.2,y:cy+1.95,w:cw-0.4,h:0.95,fontFace:BODY,fontSize:14,color:INK,align:"center",valign:"top",margin:0,lineSpacingMultiple:1.05});
  });
  s.addText("Параметры модели не настраивались под ожидаемый результат.",
    {x:M,y:5.2,w:W-2*M,h:0.8,fontFace:TB,fontSize:19,bold:true,color:STEEL,align:"center",valign:"middle"});
  s.addNotes("Три независимых проверки: точные аналитические решения, баланс энергии и реальный двигатель. Главное — числа не подгонялись под ответ.");
}

// ============================ S8 · КАРТА РИСКА ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Результат 1: карта риска по сечению магнита");
  s.addImage({ path:FIG+"fig_risk_map_slide.png", x:M-0.1, y:1.6, w:4.7, h:4.7 });
  const rx=5.3, rw=W-M-rx;
  const items=[
    ["Соседние полюса повреждены по-разному: от нетронутого до 91 % объёма — в одном и том же режиме.",RED],
    ["Средняя оценка «по худшей точке» этого не покажет — нужен расчёт поля.",STEEL],
    ["Дальше: что стало с характеристиками и где граница по нагреву.",AMBER],
  ];
  let yy=2.1;
  items.forEach(([t,c])=>{
    s.addShape(p.ShapeType.ellipse,{x:rx,y:yy+0.05,w:0.28,h:0.28,fill:{color:c}});
    s.addText(t,{x:rx+0.45,y:yy-0.1,w:rw-0.45,h:1.1,fontFace:BODY,fontSize:16.5,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.06});
    yy+=1.35;
  });
  s.addNotes("Это не схема, а результат расчёта: каждая ячейка сечения окрашена фактической долей сохранённой ремнантности. Режим один и тот же для всей машины — среда 130 градусов, ток 40 ампер, рабочая точка по оси q. Главное, что видно: соседние полюса повреждены совершенно по-разному. Два полюса остались целы, а самые нагруженные потеряли до девяноста одного процента объёма. Причина — сочетание двенадцати пазов и четырнадцати полюсов: каждый полюс стоит в своём положении относительно поля обмотки. Что это доказывает: оценка по одной «худшей точке» или по среднему такую картину не даёт в принципе — нужен расчёт поля по всему объёму. И это не численный шум: картина имеет двукратную симметрию, ровно как требует наибольший общий делитель двенадцати и четырнадцати, равный двум.");
}

// ============================ S9 · ВЫБОР МАТЕРИАЛА (KEY) ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Результат 2: обоснование выбора материала");
  s.addText("При насыщенном магнитопроводе замена Nd-Fe-B → Sm-Co:",
    {x:M,y:1.55,w:W-2*M,h:0.6,fontFace:BODY,fontSize:19,color:INK,align:"left",valign:"middle"});
  const cw=(W-2*M-0.4)/2, cy=2.35, ch=2.4;
  card(s,M,cy,cw,ch,"FDECEA");
  s.addText("−3 %",{x:M,y:cy+0.35,w:cw,h:1.1,fontFace:TB,fontSize:60,bold:true,color:HEAT,align:"center",valign:"middle",margin:0});
  s.addText("падение момента\n(в рассмотренном примере)",{x:M+0.2,y:cy+1.55,w:cw-0.4,h:0.8,fontFace:BODY,fontSize:16,color:INK,align:"center",valign:"top",margin:0,lineSpacingMultiple:1.05});
  const x2=M+cw+0.4; card(s,x2,cy,cw,ch,"E7F4EC");
  s.addText("✓",{x:x2,y:cy+0.3,w:cw,h:1.1,fontFace:TB,fontSize:54,bold:true,color:GREEN,align:"center",valign:"middle",margin:0});
  s.addText("выигрыш по температурной стойкости — сохраняется",{x:x2+0.2,y:cy+1.5,w:cw-0.4,h:0.85,fontFace:BODY,fontSize:16,color:INK,align:"center",valign:"top",margin:0,lineSpacingMultiple:1.05});
  card(s,M,5.05,W-2*M,1.2,"EAF1F7");
  s.addText("Величину потери определяет насыщение магнитопровода — то есть геометрия машины, а не только магнит.",
    {x:M+0.3,y:5.05,w:W-2*M-0.6,h:1.2,fontFace:BODY,fontSize:17,color:INK,align:"left",valign:"middle",margin:0,lineSpacingMultiple:1.05});
  s.addNotes("Главный результат для этой аудитории: в насыщенной машине переход на SmCo стоит всего около трёх процентов момента, а термостойкость сохраняется. Цена перехода определяется насыщением, то есть геометрией.");
}

// ============================ S10 · ИМПОРТОЗАМЕЩЕНИЕ ============================
{ const s=p.addSlide(); bg(s,LIGHT); title(s,"Sm-Co и импортозамещение");
  const items=[
    ["Без тяжёлых РЗМ","Sm-Co — самарий и кобальт, без диспрозия и тербия."],
    ["Отечественные марки","Инструмент подбирает наши магниты под теплонагруженные применения."],
    ["Меньше зависимости","Снижение потребности в дефицитном импортном сырье."],
  ];
  let yy=1.85; const rh=1.45;
  items.forEach(([h1,t],i)=>{
    card(s,M,yy,W-2*M,rh-0.2);
    badge(s,M+0.3,yy+0.28,String(i+1),STEEL);
    s.addText(h1,{x:M+1.25,y:yy+0.15,w:W-2*M-1.6,h:0.5,fontFace:TB,fontSize:18,bold:true,color:INK,align:"left",margin:0});
    s.addText(t,{x:M+1.25,y:yy+0.62,w:W-2*M-1.6,h:0.55,fontFace:BODY,fontSize:16,color:INK,align:"left",valign:"top",margin:0,lineSpacingMultiple:1.03});
    yy+=rh;
  });
  s.addNotes("SmCo не содержит тяжёлых редкоземельных Dy и Tb. Метод помогает подбирать отечественные магниты под конкретные теплонагруженные машины — это прямой вклад в импортозамещение.");
}

// ============================ S11 · ИТОГ + ПРИГЛАШЕНИЕ ============================
{ const s=p.addSlide(); bg(s,DARK);
  s.addText("Итог и перспективы",{x:M,y:0.5,w:W-2*M,h:0.9,fontFace:TB,fontSize:28,bold:true,color:WHITE,align:"left",valign:"middle"});
  s.addText([{text:"Сделано:  ",options:{bold:true,color:GREEN}},
    {text:"связанный магнитотепловой расчёт с необратимым размагничиванием, проверка достоверности, программный комплекс с интерфейсом.",options:{color:"DCE4EC"}}],
    {x:M,y:1.6,w:W-2*M,h:1.1,fontFace:BODY,fontSize:18,align:"left",valign:"top",lineSpacingMultiple:1.1});
  s.addText([{text:"Далее:  ",options:{bold:true,color:AMBER}},
    {text:"ускоренная нейросетевая суррогатная модель и оптимизация конструкций двигателей БПЛА.",options:{color:"DCE4EC"}}],
    {x:M,y:2.85,w:W-2*M,h:0.9,fontFace:BODY,fontSize:18,align:"left",valign:"top",lineSpacingMultiple:1.1});
  card(s,M,3.95,W-2*M,1.5,"2A3A4C");
  s.addText([{text:"Приглашаю к сотрудничеству:  ",options:{bold:true,color:HEAT}},
    {text:"нужны измеренные характеристики магнитов Sm-Co для валидации — производителям и исследователям.",options:{color:"EAF0F5"}}],
    {x:M+0.3,y:3.95,w:W-2*M-0.6,h:1.5,fontFace:BODY,fontSize:18,align:"left",valign:"middle",margin:0,lineSpacingMultiple:1.08});
  s.addText([
    {text:"Рубцов Сергей Александрович", options:{bold:true,color:WHITE,fontSize:18,breakLine:true}},
    {text:"АО «НПП «Исток» им. А. И. Шокина», г. Фрязино", options:{color:"AEBECB",fontSize:14,breakLine:true}},
    {text:"rubczov.serg@yandex.ru", options:{color:"9FE7C8",fontSize:15}},
  ],{x:M,y:5.75,w:W-2*M,h:1.3,fontFace:BODY,align:"left",valign:"top",lineSpacingMultiple:1.15});
  s.addNotes("Итог: ядро построено и проверено. Дальше — нейросетевой суррогат и оптимизация. И главное для зала: ищу измеренные SmCo для валидации, приглашаю к сотрудничеству. Спасибо за внимание.");
}

const OUT = IMG + "2026-10_magnetconf_slides_Rubtsov.pptx";
p.writeFile({ fileName: OUT }).then(f=>console.log("WROTE", f)).catch(e=>console.error("ERR", e));
