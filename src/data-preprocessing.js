var manavgat = ee.Geometry.Point([31.48, 36.88]); 
Map.centerObject(manavgat, 12); 

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var yanginSonrasi = s2.filterBounds(manavgat)
                     .filterDate('2021-08-10', '2021-08-25') 
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) 
                     .median();

var yanginParam = {bands: ['B12', 'B8', 'B4'], min: 0, max: 5000};
Map.addLayer(yanginSonrasi, yanginParam, 'Manavgat Yanık Alan Analizi');

var egitimVerisi = yanmis.merge(saglikli);
var yanmis_etiketli = yanmis.map(function(f) { return f.set('class', 1); });
var saglikli_etiketli = saglikli.map(function(f) { return f.set('class', 0); });
var egitimVerisi = yanmis_etiketli.merge(saglikli_etiketli);
var bantlar = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

print("Eğitim Verisi Sayısı:", egitimVerisi.size());
var egitimPikselleri = yanginSonrasi.select(bantlar).sampleRegions({
  collection: egitimVerisi,
  properties: ['class'],
  scale: 30,
  geometries: true
});
print("Piksel Verisi (Eşleşen):", egitimPikselleri.size());

var classifier = ee.Classifier.smileRandomForest(100).train({
  features: egitimPikselleri,
  classProperty: 'class',
  inputProperties: bantlar
});
var siniflandirilmis = yanginSonrasi.select(bantlar).classify(classifier);
Map.addLayer(siniflandirilmis, {min:0, max:1, palette: ['green', 'red']}, 'Random Forest Sonucu')

var trainAccuracy = classifier.confusionMatrix().accuracy();
print('Model Eğitim Doğruluğu:', trainAccuracy);
print('Hata Matrisi (Confusion Matrix):', classifier.confusionMatrix());

var yanmisAlan = siniflandirilmis.eq(1);
var alanMetrekare = yanmisAlan.multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: manavgat,
  scale: 10,
  maxPixels: 1e13
});
var alanHektar = ee.Number(alanMetrekare.get('classification')).divide(10000);
print('Toplam Yanan Alan (Hektar):', alanHektar);

Export.image.toDrive({
  image: siniflandirilmis,
  description: 'Manavgat_Yangin_Haritasi',
  scale: 10,
  region: manavgat,
  maxPixels: 1e13
});


