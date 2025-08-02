/**
 * Calculate the distance between two coordinates using the Haversine formula
 * @param lat1 - Latitude of first point
 * @param lon1 - Longitude of first point
 * @param lat2 - Latitude of second point
 * @param lon2 - Longitude of second point
 * @returns Distance in kilometers
 */
export function calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Radius of the Earth in kilometers
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = 
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const distance = R * c; // Distance in kilometers
  return distance;
}

/**
 * Calculate delivery days based on distance from warehouse
 * @param distance - Distance in kilometers
 * @returns Number of delivery days
 */
export function calculateDeliveryDays(distance: number): number {
  // Delivery logic based on distance
  if (distance <= 50) return 1;        // Same city - 1 day
  if (distance <= 200) return 2;       // Same state - 2 days
  if (distance <= 500) return 3;       // Nearby states - 3 days
  if (distance <= 1000) return 4;      // Medium distance - 4 days
  if (distance <= 1500) return 5;      // Far distance - 5 days
  if (distance <= 2000) return 6;      // Very far - 6 days
  return 7;                            // Maximum 7 days for anywhere in India
}

/**
 * Get estimated delivery days for a product based on user location
 * @param userLat - User's latitude
 * @param userLon - User's longitude
 * @param warehouseLat - Warehouse latitude
 * @param warehouseLon - Warehouse longitude
 * @returns Estimated delivery days
 */
export function getEstimatedDeliveryDays(
  userLat: number,
  userLon: number,
  warehouseLat: number,
  warehouseLon: number
): number {
  const distance = calculateDistance(userLat, userLon, warehouseLat, warehouseLon);
  return calculateDeliveryDays(distance);
}
