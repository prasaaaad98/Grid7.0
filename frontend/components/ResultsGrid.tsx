"use client"

import { useState, useEffect } from "react"
import { Star, Heart } from "lucide-react"
import { getEstimatedDeliveryDays } from "@/lib/deliveryUtils"

interface Product {
  id: number
  title: string
  brand: string
  category: string
  price: number
  retail_price: number
  images: string[]
  rating: number
  description: string
  assured_badge?: boolean
  isSponsored?: boolean
  warehouse_loc?: {
    name: string
    coords: [number, number]
  }
  calculatedDeliveryDays?: number
}

interface ResultsGridProps {
  query: string
  filters: any
  sort: string
  userLat: number | null
  userLon: number | null
  onLoadingChange?: (loading: boolean) => void
}

export default function ResultsGrid({ query, filters, sort, userLat, userLon, onLoadingChange }: ResultsGridProps) {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)
  
  console.log('ResultsGrid render - sort:', sort, 'query:', query)

  // Update parent loading state when our loading state changes
  const updateLoading = (isLoading: boolean) => {
    setLoading(isLoading)
    onLoadingChange?.(isLoading)
  }

  useEffect(() => {
    if (!query.trim()) {
      setProducts([])
      return
    }

    updateLoading(true)

    // Real backend call
    const searchParams = new URLSearchParams({
      q: query,
      min_price: filters.priceMin.toString(),
      max_price: filters.priceMax.toString(),
      min_rating: filters.rating.toString(),
      brand: filters.brands.join(','),
      category: filters.categories.join(','),
      sort: sort
    })

    console.log('Search params:', searchParams.toString())
    fetch(`http://localhost:8000/search?${searchParams}`)
      .then(res => res.json())
      .then(data => {
        console.log('Received data:', data.results?.slice(0, 3).map((item: any) => ({ title: item.title, price: item.price })))
        // Handle new backend response format with results, total_hits, and facets
        const products = data.results || data // Fallback to old format if needed
        const sponsoredId = data.sponsored_id // Get sponsored product ID
        
        // Transform backend data to match frontend Product interface
        let transformedProducts: Product[] = products.map((item: any, index: number) => {
          const product: Product = {
            id: item.id || index + 1,
            title: item.title,
            brand: item.brand,
            category: item.category,
            price: item.price,
            retail_price: item.retail_price,
            images: item.images,
            rating: item.rating,
            description: item.description,
            assured_badge: item.assured_badge,
            isSponsored: item.id === 1, // Simple logic - first product could be sponsored
            warehouse_loc: item.warehouse_loc
          }

          // Calculate delivery days - ONLY use coordinates, no hardcoded fallbacks
          if (item.warehouse_loc?.coords && userLat && userLon) {
            // Dynamic calculation based on distance
            const [warehouseLat, warehouseLon] = item.warehouse_loc.coords
            product.calculatedDeliveryDays = getEstimatedDeliveryDays(
              userLat, 
              userLon, 
              warehouseLat, 
              warehouseLon
            )
            console.log(`🚚 Product ${item.title}: User(${userLat.toFixed(2)}, ${userLon.toFixed(2)}) -> ${item.warehouse_loc.name}(${warehouseLat}, ${warehouseLon}) = ${product.calculatedDeliveryDays} days`)
          } else {
            // Skip products without coordinates or user location
            if (!item.warehouse_loc?.coords) {
              console.log(`❌ Product ${item.title}: No warehouse coordinates - skipping`)
              return null // Skip this product
            }
            if (!userLat || !userLon) {
              console.log(`⚠️ Product ${item.title}: No user location available yet`)
              product.calculatedDeliveryDays = 5 // Temporary placeholder until location is available
            }
          }

          return product
        }).filter(Boolean) // Remove null products)

        // Apply delivery filter
        if (filters.deliveryDays < 7) {
          transformedProducts = transformedProducts.filter(product => 
            (product.calculatedDeliveryDays || 7) <= filters.deliveryDays
          )
        }

        console.log('Setting products:', transformedProducts.slice(0, 3).map(p => ({ 
          title: p.title, 
          price: p.price, 
          deliveryDays: p.calculatedDeliveryDays 
        })))
        setProducts(transformedProducts)
      })
      .catch(error => {
        console.error('Error fetching search results:', error)
        setProducts([]) // Just set empty array on error
      })
      .finally(() => updateLoading(false))
  }, [query, filters, sort, userLat, userLon])

  const addToCart = (product: Product) => {
    const cart = JSON.parse(localStorage.getItem("cart") || "[]")
    const existingItemIndex = cart.findIndex((item: any) => item.id === product.id)

    if (existingItemIndex > -1) {
      // Item exists, increase quantity
      cart[existingItemIndex].quantity += 1
    } else {
      // New item, add to cart
      cart.push({
        id: product.id,
        title: product.title,
        price: Math.round(product.price),
        quantity: 1,
      })
    }

    localStorage.setItem("cart", JSON.stringify(cart))

    // Trigger storage event for cart count update
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify(cart),
      }),
    )

    alert(`${product.title} added to cart!`)
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-white p-4 rounded shadow animate-pulse">
              <div className="bg-gray-200 h-48 rounded mb-4"></div>
              <div className="bg-gray-200 h-4 rounded mb-2"></div>
              <div className="bg-gray-200 h-4 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!query) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>Search for products to see results</p>
      </div>
    )
  }

  return (
    <div className="p-2 sm:p-4">
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-4">
        {products.map((product, index) => {
          return (
                         <div key={`${product.id}-${index}`} className="bg-white rounded shadow hover:shadow-lg transition-shadow p-2 sm:p-4 flex flex-col h-full">
               <div className="relative mb-2 sm:mb-4">
                 <img
                   src={product.images && product.images.length > 0 ? product.images[0] : "/placeholder.svg"}
                   alt={product.title}
                   loading="lazy"
                   className="w-full h-32 sm:h-48 object-cover rounded"
                 />
                 <button className="absolute top-1 right-1 sm:top-2 sm:right-2 p-1 bg-white rounded-full shadow">
                   <Heart size={12} className="sm:w-4 sm:h-4 text-gray-400" />
                 </button>
               </div>

               <h3 className="font-medium text-xs sm:text-sm mb-1 sm:mb-2 line-clamp-2">{product.title}</h3>

               <div className="flex items-center mb-1 sm:mb-2">
                 <div className="flex items-center bg-green-500 text-white px-1 py-0.5 rounded text-xs">
                   <span className="text-xs">{product.rating}</span>
                   <Star size={8} className="sm:w-2.5 sm:h-2.5 ml-1 fill-current" />
                 </div>
                                  {product.assured_badge && (
                   <div className="flex items-center bg-blue-600 text-white px-1 py-0.5 rounded text-xs ml-2">
                     <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                       <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                     </svg>
                     Assured
                   </div>
                 )}
                 {product.isSponsored && (
                   <span className="text-xs text-gray-500 ml-2">*sponsored</span>
                 )}
               </div>

               <div className="mb-1 sm:mb-2">
                 <span className="text-sm sm:text-lg font-semibold">
                   ₹{Math.round(product.price).toLocaleString()}
                 </span>
                 {product.retail_price > product.price && (
                   <span className="text-xs sm:text-sm text-gray-500 line-through ml-1 sm:ml-2">
                     ₹{product.retail_price.toLocaleString()}
                   </span>
                 )}
               </div>

               {/* Delivery Information */}
               {product.calculatedDeliveryDays && (
                 <div className="mb-1 sm:mb-2">
                   <span className="text-xs text-green-600 font-medium">
                     📦 Delivery in {product.calculatedDeliveryDays} day{product.calculatedDeliveryDays > 1 ? 's' : ''}
                     {product.warehouse_loc?.name && (
                       <span className="text-gray-500 ml-1">from {product.warehouse_loc.name.length > 8 ? product.warehouse_loc.name.substring(0, 8) + '..' : product.warehouse_loc.name}</span>
                     )}
                   </span>
                 </div>
               )}

               <div className="space-y-1 sm:space-y-2 mt-auto">
                 <button
                   onClick={() => addToCart(product)}
                   className="w-full bg-[#ff9f00] hover:bg-[#e68900] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors"
                 >
                   Add to Cart
                 </button>
                 <button className="w-full bg-[#fb641b] hover:bg-[#e55a1b] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors">
                   Buy Now
                 </button>
               </div>
             </div>
          )
        })}
      </div>

      {products.length === 0 && query && (
        <div className="text-center py-8 text-gray-500">
          <p className="text-sm sm:text-base">No products found for "{query}"</p>
          <p className="text-xs sm:text-sm mt-2">Try searching with different keywords</p>
        </div>
      )}
    </div>
  )
}
