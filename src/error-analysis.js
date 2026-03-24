var manavgat = ee.Geometry.Point([31.48, 36.88]); 
Map.centerObject(manavgat, 12); 

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var yanginSonrasi = s2.filterBounds(manavgat)
                     .filterDate('2021-08-10', '2021-08-25') 
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) 
                     .median();

var bantlar = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

var yanmisEtiketli = yanmis.map(function(f) { return f.set('class', 1); });
var saglikliEtiketli = saglikli.map(function(f) { return f.set('class', 0); });
var suEtiketli = su.map(function(f) { return f.set('class', 2); }); // Su sınıfı eklendi

// VERSION 1 - WITHOUT WATER CLASS (Sadece Yanmış ve Sağlıklı)
var egitimVerisi_v1 = yanmisEtiketli.merge(saglikliEtiketli);

// VERSION 2 - WITH WATER CLASS (Yanmış, Sağlıklı ve Su)
var egitimVerisi_v2 = yanmisEtiketli.merge(saglikliEtiketli).merge(suEtiketli);

var pikseller_v1 = yanginSonrasi.select(bantlar).sampleRegions({
  collection: egitimVerisi_v1,
  properties: ['class'],
  scale: 30
}).randomColumn('split');

var pikseller_v2 = yanginSonrasi.select(bantlar).sampleRegions({
  collection: egitimVerisi_v2,
  properties: ['class'],
  scale: 30
}).randomColumn('split');

var egitim_v1 = pikseller_v1.filter(ee.Filter.lt('split', 0.1));
var egitim_v2 = pikseller_v2.filter(ee.Filter.lt('split', 0.1));

var svm_v1 = ee.Classifier.libsvm().train(egitim_v1, 'class', bantlar);
var svm_v2 = ee.Classifier.libsvm().train(egitim_v2, 'class', bantlar);

var test_v1 = pikseller_v1.filter(ee.Filter.gte('split', 0.7));
var test_v2 = pikseller_v2.filter(ee.Filter.gte('split', 0.7));

var svmMatrix_v1 = test_v1.classify(svm_v1).errorMatrix('class', 'classification');
var svmMatrix_v2 = test_v2.classify(svm_v2).errorMatrix('class', 'classification');

print('--- SVM V1: SU SINIFI YOK (Hatalı) ---');
print('Doğruluk Oranı:', svmMatrix_v1.accuracy());
print('Hata Matrisi:', svmMatrix_v1);

print('--- SVM V2: SU SINIFI VAR (İyileştirilmiş) ---');
print('Doğruluk Oranı:', svmMatrix_v2.accuracy());
print('Hata Matrisi:', svmMatrix_v2);

var svmHarita_v1 = yanginSonrasi.select(bantlar).classify(svm_v1);
var svmHarita_v2 = yanginSonrasi.select(bantlar).classify(svm_v2);

// Sadece yanmış alanları (class 1) görselleştiriyoruz
Map.addLayer(svmHarita_v1.updateMask(svmHarita_v1.eq(1)), {palette: ['red']}, 'SVM V1 (Su Yok - Kırmızı)');
Map.addLayer(svmHarita_v2.updateMask(svmHarita_v2.eq(1)), {palette: ['blue']}, 'SVM V2 (Su Var - Mavi)');

var alan_v1 = svmHarita_v1.eq(1).multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(), geometry: manavgat.buffer(25000).bounds(), scale: 30, maxPixels: 1e13
});
var alan_v2 = svmHarita_v2.eq(1).multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(), geometry: manavgat.buffer(25000).bounds(), scale: 30, maxPixels: 1e13
});

print('--- ALAN FARKI ---');
print('V1 Yanmış Alan (ha):', ee.Number(alan_v1.get('classification')).divide(10000));
print('V2 Yanmış Alan (ha):', ee.Number(alan_v2.get('classification')).divide(10000));
